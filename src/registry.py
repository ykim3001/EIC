PARTS = {
    "attn_qkv": ("attn", "c_attn"),
    "attn_out": ("attn", "c_proj"),
    "mlp_in": ("mlp", "c_fc"),
    "mlp_out": ("mlp", "c_proj"),
}

def resolve_bits(profile: dict, default_w=8, default_a=8, part: str = None):
    if part and part in profile:
        w = profile[part].get("w", default_w)
        a = profile[part].get("a", default_a)
    else:
        w = profile.get("default_weight_bits", default_w)
        a = profile.get("default_activation_bits", default_a)
    return int(w), int(a)