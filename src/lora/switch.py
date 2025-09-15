from peft import LoraConfig, get_peft_model
import torch.nn as nn

class MultiLoRASwitch(nn.Module):
    def __init__(self, base: nn.Module, target_modules, r=8, alpha=16, dropout=0.05,
    names=("uniform8", "mixed8648")):
        super().__init__()
        self.base = base
        self.adapters = {}
        for nm in names:
            lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none")
            self.adapters[nm] = get_peft_model(self.base, lcfg)
            self.adapters[nm].disable_adapter_layers()
        self.active = None

    def activate(self, name: str):
        if self.active:
            self.adapters[self.active].disable_adapter_layers()
        self.adapters[name].enable_adapter_layers()
        self.active = name

    def forward(self, *a, **kw):
        return self.adapters[self.active](*a, **kw)