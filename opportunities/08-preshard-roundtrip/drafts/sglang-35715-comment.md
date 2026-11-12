# Draft comment on sgl-project/sglang#35715 (or a note to its author, Youssef)

Attach `patches/sglang/0001-deepseek_v2-kv_b_proj-alias-main-95999790a4.patch` (it also applies to v0.5.20 as the `-v0.5.20` file).

---

This no longer applies as written. 2d8484740d (#35314, 26 Aug) changed the fill-in in `forward_prepare` to:

```python
if self.attn_mha.kv_b_proj is None and hasattr(self, "kv_b_proj"):
```

The underlying bug is unchanged on v0.5.20 through v0.5.23 and on current main. `__init__` still sets `self.attn_mha.kv_b_proj = None` (v0.5.20 `deepseek_v2.py:1920`), and the first forward still registers the alias as a submodule (`:2070-2071`). In any engine that captures CUDA graphs before saving, a `sharded_state` checkpoint therefore keeps only the `attn_mha.kv_b_proj.weight` name, and a fresh model fails to load it with a `KeyError`.

#35702 was auto-closed as inactive on 28 Oct, but it still reproduces. A CPU reproduction with v0.5.20's own `ShardedStateLoader.save_model`/`load_model` and these two statements shows:
- `KeyError` after a forward;
- a clean load without one;
- an exact round trip with the change below.

The attached refresh makes the same change rebased over the `hasattr` guard: alias with `object.__setattr__` in `__init__`, and drop the fill-in. It has the same 3+/3− shape as the original.

Happy to add a CPU unit test if that helps review.
