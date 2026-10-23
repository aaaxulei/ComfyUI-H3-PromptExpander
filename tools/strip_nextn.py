"""Write a copy of a qwen35 GGUF without its nextn (MTP) layer so older llama.cpp builds can load it.

Usage: python tools/strip_nextn.py in.gguf out-noMTP.gguf"""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
n_layer = int(r.fields[f"{arch}.block_count"].parts[r.fields[f"{arch}.block_count"].data[0]][0])
n_nextn = int(r.fields[f"{arch}.nextn_predict_layers"].parts[r.fields[f"{arch}.nextn_predict_layers"].data[0]][0])
n_main = n_layer - n_nextn
print(f"arch={arch} n_layer={n_layer} nextn={n_nextn} -> keeping blk.0..{n_main-1}")

w = GGUFWriter(dst, arch)
for name, field in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = field.types[0]
    if vt == GGUFValueType.ARRAY:
        inner = field.types[1]
        if inner == GGUFValueType.STRING:
            vals = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        else:
            vals = [field.parts[i].tolist()[0] for i in field.data]
        if name == f"{arch}.block_count":
            pass
        w.add_array(name, vals)
        continue
    val = field.parts[field.data[0]]
    if vt == GGUFValueType.STRING:
        w.add_string(name, bytes(val).decode("utf-8"))
    elif name == f"{arch}.block_count":
        w.add_uint32(name, n_main)
    elif name == f"{arch}.nextn_predict_layers":
        w.add_uint32(name, 0)
    elif vt == GGUFValueType.UINT32:
        w.add_uint32(name, int(val[0]))
    elif vt == GGUFValueType.INT32:
        w.add_int32(name, int(val[0]))
    elif vt == GGUFValueType.FLOAT32:
        w.add_float32(name, float(val[0]))
    elif vt == GGUFValueType.BOOL:
        w.add_bool(name, bool(val[0]))
    elif vt == GGUFValueType.UINT64:
        w.add_uint64(name, int(val[0]))
    elif vt == GGUFValueType.FLOAT64:
        w.add_float64(name, float(val[0]))
    else:
        raise SystemExit(f"unhandled field {name} type {vt}")

kept = dropped = 0
for t in r.tensors:
    if t.name.startswith("blk."):
        if int(t.name.split(".")[1]) >= n_main:
            dropped += 1
            continue
    w.add_tensor_info(t.name, list(t.data.shape), t.data.dtype, int(t.n_bytes), raw_dtype=t.tensor_type)
    kept += 1
print(f"tensors kept={kept} dropped={dropped}")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in r.tensors:
    if t.name.startswith("blk.") and int(t.name.split(".")[1]) >= n_main:
        continue
    w.write_tensor_data(t.data)
w.close()
print("written", dst)
