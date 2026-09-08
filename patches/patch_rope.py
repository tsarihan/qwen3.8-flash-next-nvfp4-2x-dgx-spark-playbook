p = "/home/tsarihan/patches/modeling_rope_utils.py"
s = open(p).read()
old = '                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)\n'
assert s.count(old) == 1, "anchor %d" % s.count(old)
new = ('                    # PATCH: setdefault evaluates its default eagerly, so this raised\n'
       '                    # AttributeError on multimodal configs where max_position_embeddings\n'
       '                    # lives on text_config, even when the key is already present.\n'
       '                    if "original_max_position_embeddings" not in self.rope_parameters:\n'
       '                        _mpe = getattr(self, "max_position_embeddings", None)\n'
       '                        if _mpe is None:\n'
       '                            _mpe = getattr(getattr(self, "text_config", None),\n'
       '                                           "max_position_embeddings", None)\n'
       '                        self.rope_parameters["original_max_position_embeddings"] = _mpe\n')
open(p, "w").write(s.replace(old, new, 1))
print("rope util patched")
