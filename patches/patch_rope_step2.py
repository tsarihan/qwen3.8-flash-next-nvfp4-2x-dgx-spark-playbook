p = "/home/tsarihan/patches/modeling_rope_utils.py"
s = open(p).read()
if "_mm_max_position_embeddings" in s:
    print("already patched"); raise SystemExit

helper = '''

def _mm_max_position_embeddings(cfg):
    """max_position_embeddings, falling back to text_config.

    On multimodal wrapper configs (e.g. Qwen4ExpConfig) the attribute lives on
    text_config, so a bare self.max_position_embeddings raises AttributeError
    on every non-default rope path.
    """
    v = getattr(cfg, "max_position_embeddings", None)
    if v is None:
        v = getattr(getattr(cfg, "text_config", None), "max_position_embeddings", None)
    return v

'''
# insert helper after the imports block
marker = "\ndef "
i = s.index(marker)
s = s[:i] + helper + s[i:]

n = s.count("self.max_position_embeddings")
s = s.replace("self.max_position_embeddings", "_mm_max_position_embeddings(self)")
# repair the one already inside our earlier patch (it used getattr, keep it simple)
s = s.replace('_mpe = getattr(self, "max_position_embeddings", None)',
              '_mpe = _mm_max_position_embeddings(self)')
open(p, "w").write(s)
print("replaced %d occurrences of self.max_position_embeddings" % n)
