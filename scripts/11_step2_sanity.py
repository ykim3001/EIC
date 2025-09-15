import os, sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transformers import AutoTokenizer
from src.modeling.gpt2_patch import GPT2QuantModel

uniform8 = {"name":"uniform8","default_weight_bits":8,"default_activation_bits":8}
mixed8648 = {"name":"mixed8648","default_weight_bits":8,"default_activation_bits":8,
             "attn_qkv":{"w":6,"a":8}, "attn_out":{"w":6,"a":8}, "mlp_in":{"w":4,"a":8}, "mlp_out":{"w":6,"a":8}}

if __name__ == "__main__":
  tok = AutoTokenizer.from_pretrained("gpt2")
  m = GPT2QuantModel("gpt2", uniform8)
  x = tok("LoRA sanity.", return_tensors="pt")
  y1 = m(**x).logits.detach().norm().item()
  m.set_bits_profile(mixed8648)
  y2 = m(**x).logits.detach().norm().item()
  print("uniform8 (LoRA-on) norm:", y1)
  print("mixed8648 (LoRA-on) norm:", y2)
