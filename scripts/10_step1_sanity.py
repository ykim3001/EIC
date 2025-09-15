import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from transformers import AutoTokenizer
from src.modeling.gpt2_patch import GPT2QuantModel

uniform8 = {"default_weight_bits": 8, "default_activation_bits": 8}
mixed8648 = {
    "default_weight_bits": 8, "default_activation_bits": 8,
    "attn_qkv": {"w": 6, "a": 8},
    "attn_out": {"w": 6, "a": 8},
    "mlp_in": {"w": 4, "a": 8},
    "mlp_out": {"w": 6, "a": 8},
}

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    m = GPT2QuantModel("distilgpt2", uniform8)
    text = "Quantization sanity test."
    inputs = tok(text, return_tensors="pt")

    out1 = m(**inputs).logits.detach()

    m.set_bits_profile(mixed8648)
    out2 = m(**inputs).logits.detach()

    print("Uniform8 logits norm:", out1.norm().item())
    print("Mixed8648 logits norm:", out2.norm().item())

    block0 = m.model.transformer.h[0]
    print("Block0 mlp.c_fc bits:", block0.mlp.c_fc.w_bits, block0.mlp.c_fc.a_bits)