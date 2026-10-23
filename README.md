# ComfyUI-H3-PromptExpander

One node, **MiniMax H3 Prompt Expander (local GGUF)**, that turns a short request plus optional
images or reference videos into a MiniMax-H3 structured prompt. It runs a local GGUF Qwen vision
model through llama-cpp-python and feeds it the official `h3-prompt-writing` guide from
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)
(`references/base-en.txt`, `references/ref-en.txt`). The *Output Rules* and *Tips for Better Results* sections of
`references/SKILL.md` are spliced into the system prompt too, so extra house rules go there. Nothing leaves the machine.

## Models

Put the language GGUF and its `mmproj-*.gguf` vision projector in `models/LLM/`.
Tested with `Qwen_Qwen3.5-9B-Q8_0.gguf` + `mmproj-Qwen_Qwen3.5-9B-bf16.gguf` and with
`Qwen3.8-27B-Heretic-ara-noMTP-Q4_K_M.gguf` + `mmproj-Qwen3.8-27B-Heretic-ara-F16.gguf`; Qwen3.8-27B GGUFs use
the same `qwen3.5 / qwen3.8` chat family. Qwen3-VL GGUFs use `qwen3-vl`.

Qwen3.8 GGUFs carry an MTP (`nextn`) layer that llama-cpp-python 0.3.35 cannot load
(`missing tensor 'blk.64.ssm_conv1d.weight'`). llama-cpp-python 0.4.3 (JamePeng, 2026-10-17) loads them; on an
older build use a `noMTP` file instead. MTP only speeds up decoding and does not change the output.

## Mode selection

| connected inputs | mode | guide |
|---|---|---|
| none | T2VA | base-en |
| first_frame | I2VA | base-en |
| first_frame + last_frame | FL2VA | base-en |
| last_frame | L2VA | base-en |
| any ref_image / ref_video | Ref2VA (six sections) | ref-en |

Reference videos are IMAGE frame batches, the same convention as the core *MiniMax H3 Reference to
Video* node; `frames_per_video` frames are sampled evenly and shown with their timestamps.

## Outputs

`prompt` is the complete text for the H3 text encoder. `description`, `soundscape` and `music` are
the three fields split out for hand edits.

`tools/strip_nextn.py in.gguf out.gguf` writes that noMTP copy from a GGUF that has the layer.
