import json, os
src = "/data/models/qwen3.8-flash-next-nvfp4/config.json"
dst = os.path.expanduser("~/patches/qwen38fn-nvfp4-config-yarn.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
d = json.load(open(src))
before = dict(d["text_config"]["rope_parameters"])
# Exactly the block from the Qwen model card
d["text_config"]["rope_parameters"] = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}
json.dump(d, open(dst, "w"), indent=1)
print("before:", json.dumps(before))
print("after :", json.dumps(d["text_config"]["rope_parameters"]))
print("wrote :", dst, os.path.getsize(dst), "bytes")
