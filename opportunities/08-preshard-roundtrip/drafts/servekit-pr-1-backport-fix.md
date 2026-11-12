# Draft PR for eth-easl/servekit (send first: small, urgent)

**Branch:** `preshard-roundtrip`, commit 1 of 2 (`patches/servekit/0001-*.patch`). It applies on main at cd8f0b8.

**Title:** fix(patches): apply the sglang#35715 backport to sglang >= 0.5.19

---

On SGLang 0.5.19 and later, including the platform's 0.5.20, `servekit launch` drops the MLA backport with a single log line. MLA models then fail to load from a presharded checkpoint again, the same way as #19:

```
[SERVEKIT] dropping sglang_35715: exit code 1
...
KeyError: 'model.layers.N.self_attn.attn_mha.kv_b_proj.weight'
```

**Cause.** SGLang 2d8484740d (#35314, 26 Aug, first released in v0.5.19) changed the `forward_prepare` fill-in in `deepseek_v2.py` to:

```python
if self.attn_mha.kv_b_proj is None and hasattr(self, "kv_b_proj"):
```

The backport's `OLD_FORWARD` anchor therefore matches 0 times. I checked v0.5.19 through v0.5.23 and main at 95999790a4: the `__init__` anchor still matches once, and the lazy alias is still there (v0.5.20 `deepseek_v2.py:1920` and `:2070-2071`). Upstream #35715 is still open, and #35702 was auto-closed as inactive.

**Fix.** Accept both spellings of the fill-in. Nothing else changes.

**Tests.**
- An old-style fixture.
- A 0.5.19-style fixture.
- The real v0.5.20 `deepseek_v2.py`, when a vendored SGLang checkout sits next to servekit; skipped otherwise.
- 12/12 in `tests/test_patches.py`.

A CPU reproduction runs SGLang 0.5.20's own `ShardedStateLoader.save_model`/`load_model` and the two MLA statements on a tiny model:
- with the old backport, the load raises `KeyError`;
- with this one, the reloaded model matches the saved one tensor for tensor.

**Not run on a GPU yet.** The Modal runbook covers DeepSeek-V2-Lite at TP=2.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
