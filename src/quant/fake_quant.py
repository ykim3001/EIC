import torch
import torch.nn as nn

class UniformAffineQuant(nn.Module):
    def __init__(self, bits: int, per_channel: bool = False, ch_axis: int = 0, observer: str = "ema"):
        super().__init__()
        assert 2 <= bits <= 8
        self.bits = bits
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.momentum = 0.9 if observer == "ema" else 0.0
        self.register_buffer("min_val", torch.tensor(0.0), persistent=False)
        self.register_buffer("max_val", torch.tensor(0.0), persistent=False)
        self.initialized = False

    def _reduce_min_max(self, x: torch.Tensor):
        if not self.per_channel:
            return x.min(), x.max()
        dims = [d for d in range(x.dim()) if d != self.ch_axis]
        mn = x.amin(dim=dims, keepdim=True)
        mx = x.amax(dim=dims, keepdim=True)
        return mn, mx

    @torch.no_grad()
    def _maybe_init_buffers(self, x: torch.Tensor):
        if self.initialized:
            return
        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.ch_axis] = x.size(self.ch_axis)
            self.min_val = torch.zeros(shape, device=x.device, dtype=x.dtype)
            self.max_val = torch.zeros(shape, device=x.device, dtype=x.dtype)
        else:
            self.min_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            self.max_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        self.initialized = False  

    @torch.no_grad()
    def _update_range(self, x: torch.Tensor):
        self._maybe_init_buffers(x)
        cur_min, cur_max = self._reduce_min_max(x)
        if not self.initialized:
            self.min_val.copy_(cur_min)
            self.max_val.copy_(cur_max)
            self.initialized = True
        else:
            self.min_val.mul_(self.momentum).add_(cur_min * (1 - self.momentum))
            self.max_val.mul_(self.momentum).add_(cur_max * (1 - self.momentum))

    @staticmethod
    def _fake_quant(x: torch.Tensor, minv: torch.Tensor, maxv: torch.Tensor, bits: int):
        qmin, qmax = 0, (1 << bits) - 1
        scale = (maxv - minv).clamp(min=1e-8) / float(qmax - qmin)
        zero_point = (qmin - minv / scale).round().clamp(qmin, qmax)
        xq = x.clamp(minv, maxv)
        q = ((xq / scale) + zero_point).round().clamp(qmin, qmax)
        return (q - zero_point) * scale

    def forward(self, x: torch.Tensor):
        if not self.initialized:
            self._update_range(x)
        elif self.training:
            self._update_range(x)

        q = self._fake_quant(x, self.min_val, self.max_val, self.bits)
        # STE
        return x + (q - x).detach()



class QuantLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True, w_bits=8, a_bits=8, w_per_channel=True):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.wq = UniformAffineQuant(bits=w_bits, per_channel=w_per_channel, ch_axis=0)
        self.aq_in = UniformAffineQuant(bits=a_bits, per_channel=False)
        self.w_bits = w_bits
        self.a_bits = a_bits

    def set_bits(self, w_bits=None, a_bits=None):
        if w_bits is not None:
            self.w_bits = int(w_bits); self.wq.bits = int(w_bits)
        if a_bits is not None:
            self.a_bits = int(a_bits); self.aq_in.bits = int(a_bits)

    def forward(self, x):
        x = self.aq_in(x)
        if not self.wq.initialized:
            with torch.no_grad():
                self.wq._update_range(self.linear.weight)
        import os
        USE_LORA = os.getenv("USE_LORA", "1") == "1"
        FORCE_WQ = os.getenv("FORCE_WQ", "0") == "1"
        if USE_LORA and self.training and not FORCE_WQ:
            return self.linear(x)
        qw = self.wq(self.linear.weight)
        return torch.nn.functional.linear(x, qw, self.linear.bias)
