import io
p = "/home/tsarihan/patches/qsa.py"
s = io.open(p, encoding="utf-8").read()
if "QSA_FP8_KV" in s:
    print("already patched"); raise SystemExit

# make os available
if "\nimport os\n" not in s:
    s = s.replace("import torch\n", "import os\nimport torch\n", 1)

hdr = ('\n_QSA_FP8_KV = os.environ.get("QSA_FP8_KV", "0") == "1"\n'
       '_QSA_ALLOWED_KV = ("auto", "bfloat16") + (("fp8", "fp8_e4m3") if _QSA_FP8_KV else ())\n')
# insert helper after imports (before first class/def at column 0)
i = s.index("\nclass ")
s = s[:i] + hdr + s[i:]

n = 0
# 1. ClassVar list
old1 = 'supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]'
if old1 in s:
    s = s.replace(old1,
      'supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(_QSA_ALLOWED_KV)  # QSA_FP8_KV widens this', 1); n += 1

# 2 & 3. the two membership gates
old2 = 'if self.kv_cache_dtype not in ("auto", "bfloat16"):'
if old2 in s:
    s = s.replace(old2, 'if self.kv_cache_dtype not in _QSA_ALLOWED_KV:', 1); n += 1
old3 = 'if cache_config.cache_dtype not in ("auto", "bfloat16"):'
if old3 in s:
    s = s.replace(old3, 'if cache_config.cache_dtype not in _QSA_ALLOWED_KV:', 1); n += 1

# 4. the torch-dtype gate
old4 = ('        if self.kv_cache_torch_dtype != torch.bfloat16:\n'
        '            raise NotImplementedError(\n'
        '                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"\n'
        '            )\n')
new4 = ('        if self.kv_cache_torch_dtype != torch.bfloat16 and not _QSA_FP8_KV:\n'
        '            raise NotImplementedError(\n'
        '                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"\n'
        '            )\n')
if old4 in s:
    s = s.replace(old4, new4, 1); n += 1

io.open(p, "w", encoding="utf-8").write(s)
print("qsa.py patched, gates widened:", n)
