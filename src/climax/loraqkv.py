import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(0.1)

        self.num_heads = 16
        head_dim = 1024 // self.num_heads
        self.scale = head_dim ** -0.5
        

    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_c = qkv.clone()
        # qkv_ = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv_.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        
        # qkv = self.qkv(x)  # B,N,N,3*org_C
        ql = qkv[:, :, : self.dim]
        vl = qkv[:, :, -self.dim:]
        new_q = self.linear_b_q(self.linear_a_q(ql))
        new_v = self.linear_b_v(self.linear_a_v(vl))

        qkv_c[:, :, : self.dim] += new_q
        qkv_c[:, :, -self.dim:] += new_v

        qkv_c = qkv_c.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_c.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # commented before flash attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        #     attn = F.scaled_dot_product_attention(q,k,v)
            
        # x = attn.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        # qkv = self.attn_drop(qkv)
        # # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # qkv = self.proj(qkv)
        # qkv = self.proj_drop(qkv)

        return x
