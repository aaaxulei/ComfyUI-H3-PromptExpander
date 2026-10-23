"""MiniMax-H3 prompt expander on a local GGUF Qwen VLM.

The rewriting rules are the official h3-prompt-writing skill shipped in
MiniMax-AI/MiniMax-H3 (references/base-en.txt for T2VA/I2VA/FL2VA/L2VA,
references/ref-en.txt for Ref2VA). The model only has to follow them.
"""

import base64
import gc
import io as pyio
import logging
import math
import os
import re

import numpy as np
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen35ChatHandler, Qwen3VLChatHandler

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io

REFERENCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")

with open(os.path.join(REFERENCES_DIR, "base-en.txt"), encoding="utf-8") as f:
    BASE_GUIDE = f.read()
with open(os.path.join(REFERENCES_DIR, "ref-en.txt"), encoding="utf-8") as f:
    REF_GUIDE = f.read()
with open(os.path.join(REFERENCES_DIR, "SKILL.md"), encoding="utf-8") as f:
    skill = f.read()
SKILL_RULES = skill[skill.index("## Output Rules"):].strip()  # Output Rules + Tips for Better Results

if "LLM" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], {".gguf"})

SYSTEM_HEADER = (
    "You are the prompt rewriter for MiniMax-H3, a joint video and audio generation model. "
    "Rewrite the user's request into the final H3 prompt by following the guide below exactly: "
    "keep its field names, section order, labels, timing notation and vocabulary. "
    "Write in English, but keep dialogue, lyrics and visible on-screen text in their original language. "
    "Every detail must be something visible or audible; avoid abstract words. "
    "Match the total timeline to the requested duration. "
    "Output only the final prompt text: no title, no markdown, no code fence, no explanation.\n\n"
    + SKILL_RULES + "\n\n"
    "# Guide\n\n"
)

ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}

MODEL_FAMILIES = {
    "qwen3.5 / qwen3.8": Qwen35ChatHandler,
    "qwen3-vl": Qwen3VLChatHandler,
}


class _LoadedModel:
    key = None
    llm = None
    handler = None


def _unload():
    if _LoadedModel.llm is not None:
        _LoadedModel.llm.close()  # also closes the chat handler's mtmd context
    _LoadedModel.key = None
    _LoadedModel.llm = None
    _LoadedModel.handler = None
    gc.collect()
    comfy.model_management.soft_empty_cache()


def _load(model, mmproj, family, n_ctx, image_max_tokens, thinking):
    key = (model, mmproj, family, n_ctx, image_max_tokens, thinking)
    if _LoadedModel.key == key:
        return _LoadedModel.llm
    _unload()
    model_path = folder_paths.get_full_path_or_raise("LLM", model)
    mmproj_path = folder_paths.get_full_path_or_raise("LLM", mmproj)
    handler_cls = MODEL_FAMILIES[family]
    # add_vision_id off: the labels are written by us so video frames do not count as pictures
    kwargs = {"clip_model_path": mmproj_path, "verbose": False, "image_max_tokens": image_max_tokens, "add_vision_id": False}
    if handler_cls is Qwen35ChatHandler:
        kwargs["enable_thinking"] = thinking
    else:
        kwargs["force_reasoning"] = thinking
    # llama.cpp allocates outside ComfyUI's model manager, so make room for the weights, the projector and the KV cache first
    needed = os.path.getsize(model_path) + os.path.getsize(mmproj_path) + n_ctx * 256 * 1024 + 2 * 1024**3
    comfy.model_management.free_memory(needed, comfy.model_management.get_torch_device())
    logging.info("H3PromptExpander: loading %s with %s", model, mmproj)
    handler = handler_cls(**kwargs)
    llm = Llama(model_path, chat_handler=handler, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)
    _LoadedModel.key = key
    _LoadedModel.llm = llm
    _LoadedModel.handler = handler
    return llm


def _image_data_url(frame, max_size):
    arr = np.clip(frame.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    scale = min(1.0, max_size / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
    buf = pyio.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ratio_name(width, height):
    r = width / height
    return min(ASPECT_RATIOS, key=lambda k: abs(math.log(ASPECT_RATIOS[k] / r)))


def _sample_indices(count, max_frames):
    if count <= max_frames:
        return list(range(count))
    return [round(i * (count - 1) / (max_frames - 1)) for i in range(max_frames)]


FIELD_RE = re.compile(r"^(integrated_multimodal_description|detailed_description|overall_soundscape|non_diegetic_music)\s*:\s*", re.M)


def _clean_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```[a-z]*\n|\n```$", "", text).strip()
    return text


def _split_fields(text):
    fields = {}
    matches = list(FIELD_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[m.group(1)] = text[m.end():end].strip()
    description = fields.get("integrated_multimodal_description", fields.get("detailed_description", ""))
    return description, fields.get("overall_soundscape", ""), fields.get("non_diegetic_music", "")


class H3PromptExpander(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        ggufs = [f for f in folder_paths.get_filename_list("LLM") if f.lower().endswith(".gguf")]
        models = [f for f in ggufs if "mmproj" not in os.path.basename(f).lower()]
        mmprojs = [f for f in ggufs if "mmproj" in os.path.basename(f).lower()]
        return io.Schema(
            node_id="H3PromptExpander",
            display_name="MiniMax H3 Prompt Expander (local GGUF)",
            category="conditioning/video_models",
            description="Expands a short request into a MiniMax-H3 structured prompt with a local GGUF Qwen vision model, "
                        "following the official h3-prompt-writing guide. The mode (T2VA, I2VA, FL2VA, L2VA or Ref2VA) "
                        "follows from which images are connected. Reference videos are IMAGE frame batches, as on the "
                        "MiniMax H3 Reference to Video node.",
            inputs=[
                io.Combo.Input("model", options=models, tooltip="GGUF language model in models/LLM."),
                io.Combo.Input("mmproj", options=mmprojs, tooltip="Matching mmproj vision projector GGUF in models/LLM."),
                io.Combo.Input("model_family", options=list(MODEL_FAMILIES), default="qwen3.5 / qwen3.8",
                               tooltip="Chat template family of the GGUF."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, tooltip="Short description of the wanted video."),
                io.Float.Input("duration", default=5.0, min=1.0, max=60.0, step=0.5, tooltip="Target video length in seconds; H3 is trained for 4-15 s."),
                io.Combo.Input("aspect_ratio", options=["auto"] + list(ASPECT_RATIOS), default="auto",
                               tooltip="auto takes the ratio of the first frame, last frame or first reference image, else 16:9."),
                io.Image.Input("first_frame", optional=True, tooltip="First frame -> I2VA, with last_frame -> FL2VA."),
                io.Image.Input("last_frame", optional=True, tooltip="Last frame -> L2VA, with first_frame -> FL2VA."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image <Picture N> for Ref2VA."),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video <Video N> for Ref2VA, as a frame batch."),
                        prefix="ref_video_", min=0, max=3)),
                io.Int.Input("video_fps", default=24, min=1, max=120, advanced=True, tooltip="Frame rate of the reference video frame batches, for the timestamps shown to the model."),
                io.Int.Input("frames_per_video", default=8, min=1, max=32, advanced=True, tooltip="Frames sampled evenly from each reference video."),
                io.Int.Input("image_max_size", default=1024, min=64, max=4096, step=64, advanced=True, tooltip="Images are downscaled to this longest edge before the model sees them."),
                io.Int.Input("image_max_tokens", default=1024, min=64, max=16384, step=64, advanced=True, tooltip="Vision token budget per image."),
                io.Int.Input("n_ctx", default=16384, min=2048, max=262144, step=1024, advanced=True, tooltip="Context length. Each image costs up to image_max_tokens; the guide itself is about 5k tokens."),
                io.Float.Input("temperature", default=0.3, min=0.0, max=2.0, step=0.05, tooltip="0 is greedy and deterministic."),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=64, advanced=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Boolean.Input("thinking", default=False, advanced=True, tooltip="Let the model reason before writing. Slower."),
                io.Boolean.Input("keep_model_loaded", default=False, tooltip="Keep the LLM in VRAM between runs. Off frees it for the video model."),
            ],
            outputs=[
                io.String.Output(display_name="prompt", tooltip="Complete H3 prompt, ready for the text encoder."),
                io.String.Output(display_name="description", tooltip="integrated_multimodal_description (detailed_description for Ref2VA)."),
                io.String.Output(display_name="soundscape", tooltip="overall_soundscape"),
                io.String.Output(display_name="music", tooltip="non_diegetic_music"),
            ],
        )

    @classmethod
    def execute(cls, model, mmproj, model_family, prompt, duration, aspect_ratio, first_frame=None, last_frame=None,
                ref_images=None, ref_videos=None, video_fps=24, frames_per_video=8, image_max_size=1024,
                image_max_tokens=1024, n_ctx=16384, temperature=0.3, max_tokens=2048, seed=0, thinking=False,
                keep_model_loaded=False) -> io.NodeOutput:
        ref_images = [img for img in (ref_images or {}).values() if img is not None]
        ref_videos = [vid for vid in (ref_videos or {}).values() if vid is not None]

        content = []
        labels = []
        if ref_images or ref_videos:
            mode = "Ref2VA"
            guide = REF_GUIDE
            keyframes = [("first frame", first_frame), ("last frame", last_frame)]
            pictures = [(role, img) for role, img in keyframes if img is not None] + [("reference image", img) for img in ref_images]
            for n, (role, img) in enumerate(pictures, 1):
                labels.append(f"<Picture {n}>: {role}")
                content.append({"type": "text", "text": f"<Picture {n}> ({role}):"})
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(img[0], image_max_size)}})
            for n, frames in enumerate(ref_videos, 1):
                idx = _sample_indices(frames.shape[0], frames_per_video)
                labels.append(f"<Video {n}>: reference video, {frames.shape[0] / video_fps:.2f} s, {len(idx)} frames shown")
                for k in idx:
                    content.append({"type": "text", "text": f"<Video {n}> frame at {k / video_fps:.2f} s:"})
                    content.append({"type": "image_url", "image_url": {"url": _image_data_url(frames[k], image_max_size)}})
        else:
            guide = BASE_GUIDE
            if first_frame is not None and last_frame is not None:
                mode = "FL2VA"
                pictures = [("first frame of the video, 0.00 s", first_frame), (f"last frame of the video, {duration:.2f} s", last_frame)]
            elif first_frame is not None:
                mode = "I2VA"
                pictures = [("first frame of the video, 0.00 s", first_frame)]
            elif last_frame is not None:
                mode = "L2VA"
                pictures = [(f"last frame of the video, {duration:.2f} s", last_frame)]
            else:
                mode = "T2VA"
                pictures = []
            for n, (role, img) in enumerate(pictures, 1):
                labels.append(f"<Picture {n}>: {role}")
                content.append({"type": "text", "text": f"<Picture {n}> ({role}):"})
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(img[0], image_max_size)}})

        if aspect_ratio == "auto":
            anchors = [img for img in (first_frame, last_frame, *ref_images) if img is not None]
            aspect_ratio = _ratio_name(anchors[0].shape[2], anchors[0].shape[1]) if anchors else "16:9"

        task = [f"Task mode: {mode}", f"Requested duration: {duration:.2f} seconds", f"Aspect ratio: {aspect_ratio}"]
        if labels:
            task.append("Attached references, in order:\n" + "\n".join(labels))
        if mode == "Ref2VA":
            task.append("The reference videos are shown as sampled frames and are visual references for reference generation, "
                        "not source videos to edit or continue unless the user request says so. No audio is attached, so do not "
                        "define any <Audio N>. Give (Sx) speaker IDs only to subjects that actually speak or sing.")
        task.append("User request:\n" + prompt.strip())
        task.append(f"Write the final {mode} prompt now, exactly in the guide's final structure.")
        content.append({"type": "text", "text": "\n\n".join(task)})

        messages = [
            {"role": "system", "content": SYSTEM_HEADER + guide},
            {"role": "user", "content": content},
        ]

        llm = _load(model, mmproj, model_family, n_ctx, image_max_tokens, thinking)
        try:
            result = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.05,
                max_tokens=max_tokens,
                seed=seed & 0xffffffff,
            )
        finally:
            if not keep_model_loaded:
                _unload()

        text = _clean_output(result["choices"][0]["message"]["content"])
        logging.info("H3PromptExpander: %s, %s completion tokens", mode, result["usage"]["completion_tokens"])
        description, soundscape, music = _split_fields(text)
        return io.NodeOutput(text, description, soundscape, music)


class H3PromptExpanderExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3PromptExpander]


async def comfy_entrypoint() -> H3PromptExpanderExtension:
    return H3PromptExpanderExtension()
