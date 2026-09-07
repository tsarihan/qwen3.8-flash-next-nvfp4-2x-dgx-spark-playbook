p = "/home/tsarihan/patches/ple_layer.py"
src = open(p).read()

if "\nimport os\n" not in src:
    src = src.replace("import math\n", "import math\nimport os\n", 1)

old = '    if not isinstance(quant_config, Fp8Config):\n        return None\n'
new = (
    '    if not isinstance(quant_config, Fp8Config):\n'
    '        # Mixed-precision checkpoints can ship FP8-serialised PLE embedding\n'
    '        # shards even when the routed experts are NVFP4: ModelOpt excludes\n'
    '        # "*.ple.*" from NVFP4 and inherits the FP8 embedding tables from the\n'
    '        # base checkpoint. vLLM\'s ModelOpt path does not quantize embeddings\n'
    '        # at all, so the module would be built as a plain VocabParallelEmbedding\n'
    '        # and weight loading dies on ngram_embedding.weight_scale.\n'
    '        # Opt-in and reversible via env var.\n'
    '        if quant_config is not None and os.environ.get("VLLM_QWEN38FN_PLE_FP8", "0") == "1":\n'
    '            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n'
    '        return None\n'
)
n = src.count(old)
assert n == 1, "gate pattern found %d times, expected 1" % n
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patch applied")
