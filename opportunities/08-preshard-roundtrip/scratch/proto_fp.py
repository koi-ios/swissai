import torch
from torch.utils._pytree import tree_map, tree_flatten

class FT(torch.Tensor):
    @staticmethod
    def __new__(cls, inner, rec):
        r = torch.Tensor._make_wrapper_subclass(cls, inner.size(), strides=inner.stride(), storage_offset=inner.storage_offset(), dtype=inner.dtype, device=inner.device, requires_grad=False)
        r._inner = inner; r._rec = rec
        return r
    __torch_function__ = torch._C._disabled_torch_function_impl
    def __repr__(self): return f"FT({tuple(self.shape)})"
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rec = None
        def unwrap(x):
            nonlocal rec
            if isinstance(x, FT):
                rec = x._rec
                return x._inner
            return x
        uargs = tree_map(unwrap, args); ukw = tree_map(unwrap, kwargs)
        out = func(*uargs, **ukw)
        if func.is_view:
            return tree_map(lambda o: FT(o, rec) if isinstance(o, torch.Tensor) else o, out)
        # record reads
        for x in tree_flatten((args, kwargs))[0]:
            if isinstance(x, FT):
                rec.append((str(func), x._inner.storage_offset(), tuple(x._inner.size()), tuple(x._inner.stride()), x._inner.dtype))
        return out

rec=[]
base = torch.arange(24, dtype=torch.float32).reshape(4,6)
t = FT(base, rec)
param = torch.zeros(4,3)
v = t.narrow(1, 3, 3)
print(type(v), v.shape, v.stride())
param.copy_(v)
print(param)
w = t.view(torch.int32)  # dtype view
u = t.t().contiguous()
z = torch.cat([t[0:1], t[2:3]])
c = t.chunk(2, dim=0)[1].to(torch.bfloat16)
s = t[1, 2].item()
print("item", s)
for r in rec: print(r)
print("shape ok", t.shape, t.dtype, t.device, t.is_contiguous(), t.dim(), t.numel())
try:
    print("data_ptr", t.data_ptr())
except Exception as e: print("data_ptr err", type(e).__name__, e)
try:
    print("numpy", t.numpy()[0,:2])
except Exception as e: print("numpy err", type(e).__name__, str(e)[:100])
try:
    print("tolist", t[0].tolist())
except Exception as e: print("tolist err", type(e).__name__, str(e)[:100])
print("data attr type", type(t.data))
with torch.inference_mode():
    p2 = torch.empty(4,3); p2.copy_(t.narrow(1,0,3)); print("inference ok", p2[0])
