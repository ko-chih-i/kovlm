import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def depth_rope_encoding(img_features, depth_features):
    orig_dtype = img_features.dtype
    depth_features = torch.cat(depth_features,dim=0)
    depth_features = depth_features.reshape(depth_features.shape[0],-1)
    B, L, dim = img_features.shape
    assert dim % 2 == 0, "wrong dim"

    theta = 10000.0
    seqlen = 10000
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / (dim)))
    seq = torch.arange(seqlen, device=inv_freq.device, dtype=inv_freq.dtype)
    freqs = torch.outer(seq, inv_freq)

    depth_features = depth_features.clone()
    depth_features = (depth_features * seqlen).to(torch.long)

    rotary_pos_emb = freqs[depth_features]
    rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos = rotary_pos_emb.cos()
    sin = rotary_pos_emb.sin()
    img_features = (img_features * cos) + (rotate_half(img_features) * sin)
    img_features = img_features.to(orig_dtype)

    return img_features

if __name__ == "__main__":
    img_features = torch.rand(2, 10, 16)  # Example image features
    depth_features = [torch.rand(1, 10), torch.rand(1, 10)]  # Example depth features
    x = depth_rope_encoding(img_features, depth_features)