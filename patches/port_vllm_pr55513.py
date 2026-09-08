import io, re

# ---------- modelopt.py ----------
p = "/home/tsarihan/patches/modelopt.py"
s = io.open(p, encoding="utf-8").read()
if "_BLOCK_FP8_MOE_ALGOS" in s:
    print("modelopt already ported")
else:
    # 1. import Fp8Config / Fp8MoEMethod
    anchor = "from vllm.model_executor.layers.quantization.utils.fp8_utils import ("
    assert s.count(anchor) == 1
    s = s.replace(anchor,
        "from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod\n" + anchor, 1)

    # 2. block-FP8 algo names (PR 55513)
    a2 = '    # MIXED_PRECISION,\n    "MIXED_PRECISION",\n]\n'
    assert s.count(a2) == 1
    s = s.replace(a2, a2 +
        '\n# PR 55513: ModelOpt\'s canonical 2D block-FP8 name is FP8_PB_WO. Early composed\n'
        '# Qwen3.8-Flash-Next checkpoints used FP8_BLOCK_SCALES for the same tensor layout,\n'
        '# so accept both. NVIDIA\'s Qwen3.8-Flash-Next-NVFP4 declares the MTP experts as\n'
        '# {"quant_algo": "FP8_PB_WO", "group_size": 128}.\n'
        '_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")\n', 1)

    # 3. handle block-FP8 MoE in the mixed-precision RoutedExperts branch
    a3 = "        if isinstance(layer, RoutedExperts):\n"
    idx = s.rindex(a3)
    ins = ('        if isinstance(layer, RoutedExperts):\n'
           '            # PR 55513: block-FP8 MoE (e.g. the FP8 MTP experts in a mixed\n'
           '            # NVFP4 checkpoint). Built lazily so __init__ stays untouched.\n'
           '            if quant_algo in _BLOCK_FP8_MOE_ALGOS:\n'
           '                if getattr(self, "_fp8_block_config", None) is None:\n'
           '                    _bs = 128\n'
           '                    for _li in (self.quantized_layers or {}).values():\n'
           '                        if _li.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS:\n'
           '                            _bs = int(_li.get("group_size", 128))\n'
           '                            break\n'
           '                    self._fp8_block_config = Fp8Config(\n'
           '                        is_checkpoint_fp8_serialized=True,\n'
           '                        activation_scheme="dynamic",\n'
           '                        weight_block_size=[_bs, _bs],\n'
           '                    )\n'
           '                return Fp8MoEMethod(self._fp8_block_config, layer)\n')
    s = s[:idx] + ins + s[idx + len(a3):]
    io.open(p, "w", encoding="utf-8").write(s)
    print("modelopt.py ported")

# ---------- mtp.py ----------
p2 = "/home/tsarihan/patches/mtp.py"
t = io.open(p2, encoding="utf-8").read()
if "_remap_quantized_layers" in t:
    print("mtp already ported")
else:
    a = "def _make_draft_vllm_config("
    assert t.count(a) == 1
    helper = ('def _remap_quantized_layers(\n'
              '    quantized_layers: dict,\n'
              '    mtp_start_layer_idx: int,\n'
              ') -> dict:\n'
              '    """PR 55513: map checkpoint MTP layer indices to standalone draft indices."""\n'
              '    return {\n'
              '        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info\n'
              '        for name, layer_info in quantized_layers.items()\n'
              '    }\n\n\n')
    t = t.replace(a, helper + a, 1)

    a2 = ('        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)\n')
    assert t.count(a2) == 1
    ins2 = ('        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)\n'
            '        if quantized_layers:\n'
            '            setattr(  # noqa: B010\n'
            '                draft_quant_config,\n'
            '                "quantized_layers",\n'
            '                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),\n'
            '            )\n')
    t = t.replace(a2, ins2 + a2, 1)
    io.open(p2, "w", encoding="utf-8").write(t)
    print("mtp.py ported")
