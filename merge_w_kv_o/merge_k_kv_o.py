import torch

T=64
nhead=128
TP=1
kv_lora_rank=512
v_head_dim=128
hidden_size=7168
DEV='cuda'
# DTYPE=torch.bfloat16
DTYPE=torch.float

x=torch.randn(T, nhead//TP, kv_lora_rank, device=DEV, dtype=DTYPE)
w_vc=torch.randn(nhead//TP, kv_lora_rank, v_head_dim, device=DEV, dtype=DTYPE)
o_proj=torch.randn(nhead//TP, v_head_dim, hidden_size, device=DEV, dtype=DTYPE)

# standard approach
def proj_standard(x_, w_vc_, o_proj_):
    def mm_0():
        b, m, n, k = nhead//TP, T, v_head_dim, kv_lora_rank
        return torch.einsum('mbk,bkn->mbn', x_.reshape(m, b, k), w_vc_.reshape(b, k, n))

    # mm_1
    tmp_0 = mm_0()
    m, n, k = T, hidden_size, nhead//TP * v_head_dim
    o = torch.einsum('mk,kn->mn', tmp_0.reshape(m, -1), o_proj_.reshape(-1, n))
    return o

def proj_merged(x_, w_vc_, o_proj_):
    def merge_w_vc_o():
        b, m, n, k = nhead//TP, kv_lora_rank, hidden_size, v_head_dim
        return torch.einsum('bmk,bkn->bmn', w_vc_.reshape(b, m, k), o_proj_.reshape(b, k, n))
    
    # mm_1
    merged_w = merge_w_vc_o()
    m, n, k = T, hidden_size, nhead//TP * kv_lora_rank
    o = torch.einsum('mk,kn->mn', x_.reshape(m, -1), merged_w.reshape(-1, n))
    return o

o_0 = proj_standard(x, w_vc, o_proj)
o_1 = proj_merged(x, w_vc, o_proj)

print(o_0)
print(o_1)

# TODO: randn is quite big, absolute error 0.1
assert torch.allclose(o_0, o_1, rtol=1e-4, atol=1e-1)
