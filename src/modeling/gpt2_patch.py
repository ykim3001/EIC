import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import Conv1D
import os
USE_LORA = os.getenv("USE_LORA", "1") == "1"

from ..quant.fake_quant import QuantLinear
from ..registry import PARTS, resolve_bits

try:
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraModel
    PEFT_AVAILABLE = USE_LORA
except Exception:
    PEFT_AVAILABLE = False


class GPT2QuantModel(nn.Module):
    def __init__(self, model_name: str, profile: dict,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_target_modules=("linear",)):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self._wrap_all_linears()  

        self._peft_inited = False
        self._lora_cfg = dict(r=lora_r, lora_alpha=lora_alpha,
                              lora_dropout=lora_dropout,
                              target_modules=list(lora_target_modules),
                              bias="none")
        self._known_adapters = set()
        self._active_adapter = None  

        self.set_bits_profile(profile)
        self._unfreeze_all_base_params()

    def _to_quant(self, module: nn.Module) -> QuantLinear:
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            qlin = QuantLinear(in_f, out_f, bias=(module.bias is not None))
            with torch.no_grad():
                qlin.linear.weight.copy_(module.weight)
                if module.bias is not None:
                    qlin.linear.bias.copy_(module.bias)
            return qlin

        if isinstance(module, Conv1D):
            in_f, out_f = module.weight.shape
            qlin = QuantLinear(in_f, out_f, bias=(module.bias is not None))
            with torch.no_grad():
                qlin.linear.weight.copy_(module.weight.t())
                if module.bias is not None:
                    qlin.linear.bias.copy_(module.bias)
            return qlin

        raise TypeError(f"Unsupported module type for quant wrapping: {type(module)}")

    def _wrap_all_linears(self):
        for block in self.model.transformer.h:
            block.attn.c_attn = self._to_quant(block.attn.c_attn)
            block.attn.c_proj = self._to_quant(block.attn.c_proj)
            block.mlp.c_fc    = self._to_quant(block.mlp.c_fc)
            block.mlp.c_proj  = self._to_quant(block.mlp.c_proj)

    @torch.no_grad()
    def _apply_bits_profile(self, profile: dict):
        for block in self.model.transformer.h:
            for part, (submod, name) in PARTS.items():
                w_bits, a_bits = resolve_bits(profile, part=part)
                getattr(getattr(block, submod), name).set_bits(w_bits, a_bits)

    def _unfreeze_all_base_params(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def _freeze_base_except_lora(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = ("lora_" in n)


    def _ensure_peft(self):
        if not PEFT_AVAILABLE:
            return  

        if self._peft_inited:
            return
        
        lcfg = LoraConfig(**self._lora_cfg)
        self.model = get_peft_model(self.model, lcfg)  
        self._peft_inited = True
        self._known_adapters.add("default")
        self._active_adapter = "default"
        self._freeze_base_except_lora() 

    def _ensure_adapter(self, name: str):
        if not PEFT_AVAILABLE:
            return  

        self._ensure_peft()
        if name not in self._known_adapters:
            from peft import LoraConfig
            lcfg = LoraConfig(**self._lora_cfg)
            self.model.add_adapter(name, lcfg)
            self._known_adapters.add(name)
        self.model.set_adapter(name)
        self._active_adapter = name
        self._freeze_base_except_lora() 

    @torch.no_grad()
    def set_bits_profile(self, profile: dict):
        # apply quantization bits runtime
        self._apply_bits_profile(profile)

        # switch LoRA adapter to match the profile 
        pname = str(profile.get("name", "default"))
        if PEFT_AVAILABLE:
            self._ensure_adapter(pname)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
