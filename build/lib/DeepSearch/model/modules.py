import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 
from typing import Optional
from collections import OrderedDict
import einops


def stable_softmax(logits: torch.Tensor, dim=-1):
    if logits.dtype == torch.float:
        output = F.softmax(logits, dim=dim)
    elif logits.dtype == torch.bfloat16:
        output = F.softmax(logits.float(), dim=dim).bfloat16()
    elif logits.dtype == torch.half:
        output = F.softmax(logits.float(), dim=dim).half()
    else:
        raise NotImplementedError(
            "Mixed precision other than bf16 is not supported")
    return output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len,):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = d_model

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 **
                                          ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + \
            torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class NaiveMZPositionalEmbedding(nn.Module):
    def __init__(self, dim, min_wavelength=0.001, max_wavelength=2000.0) -> None:
        super().__init__()
        n_sin = dim // 2
        n_cos = dim - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, x):
        """forward _summary_

        Args:
            x (tensor): spectra with size of [B, N, 2], index 0 in third dim is m/z

        Returns:
            _type_: positional embedding with size of [B, N, dim]
        """
        x = x[:, :, [0]]
        sin_mz = torch.sin(x / self.sin_term)
        cos_mz = torch.cos(x / self.cos_term)
        return torch.cat([sin_mz, cos_mz], dim=-1)



class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, dim = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = torch.matmul(q, k_t) / math.sqrt(dim)  # scaled dot product

        # 2. apply masking (opt)
        # broadcasting
        if mask is not None:
            score.masked_fill_(mask == 0, float("-inf"))

        # 3. pass them softmax to make [0, 1] range
        attn = self.softmax(score)

        # 4. multiply with Value
        out = torch.matmul(attn, v)

        return out,  attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, dropout, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_concat = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn. Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        out = self.dropout(out)
        return out

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length,
                             self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(
            batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, d_model, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GEGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x, gate):
        """forward GEGLU forward

        Args:
            x (_type_): x 
            gate (_type_): gate

        Returns:
            _type_: _description_
        """
        x = x * self.gelu(gate)
        return x


class PositionwiseFeedFowardGeLU(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1) -> None:
        super(PositionwiseFeedFowardGeLU, self).__init__()

        self.linear1 = nn.Linear(d_model, hidden, bias=False)
        self.linear2 = nn.Linear(d_model, hidden, bias=False)
        self.linear3 = nn.Linear(hidden, d_model, bias=False)

        self.geglu = GEGLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """forward _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        gate = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(self.geglu(x, gate))
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            d_model,
            qkv_dim=None,
            num_heads=8,
            attn_drop=0.,
            proj_drop=0.,
            k_dim=None,
            v_dim=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, 'dim should be divisible by num_heads'
        self.d_model = d_model
        self.n_heads = num_heads
        self.dim_head = d_model // num_heads
        self.scale = self.dim_head ** -0.5
        self.qkv_dim = d_model if qkv_dim is None else qkv_dim 
        self.k_dim = d_model if k_dim is None else k_dim
        self.v_dim = d_model if v_dim is None else v_dim
        self.q_linear = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.k_linear = nn.Linear(self.k_dim, self.qkv_dim, bias=False)
        self.v_linear = nn.Linear(self.v_dim, self.qkv_dim, bias=False)
        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original

        self.attn_drop = nn.Dropout(attn_drop)
        
        self.out_proj = nn.Linear(self.qkv_dim, d_model)
        self.out_drop = nn.Dropout(proj_drop)


    def _init_parameters(self):
        pass


    def forward(self, 
                q, 
                memory: Optional[torch.Tensor]=None, 
                mask: Optional[torch.Tensor]=None, 
                memory_mask: Optional[torch.Tensor]=None,
                attn_bias: Optional[torch.Tensor]=None):

        has_memory = True
        device = q.device
        if memory is None:
            memory = q
            has_memory = False
        n_query, n_key = q.shape[-2], memory.shape[-2]

        q = self.q_linear(q)
        k = self.k_linear(memory)
        v = self.v_linear(memory)

        q = q * self.scale
        
        
        # split heads -> [B, H, N, D]
        q = einops.rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = einops.rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = einops.rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)

        # [B, H, N_query, N_KEY]
        qk_dot = einops.einsum(q, k, 'b h i d, b h j d -> b h i j')
        mask_value = -torch.finfo(qk_dot.dtype).max
        if attn_bias is not None:
            qk_dot += attn_bias

        if mask is not None:
            # mask should have the shape [B, N, N] for masked self attn, 
            # [B, N_query] for cross attn and self attn

            if len(mask.shape) == 3:
                qk_dot.masked_fill_(mask.unsqueeze(1), mask_value)
                #print(qk_dot)
                
            else: 
                qk_dot.masked_fill_(mask[:, None, : None], mask_value)
                
        if has_memory:
            # memory_mask have the shape [B, N_key]
            if memory_mask is not None:
                #print(memory_mask)
                memory_mask = memory_mask[:, None, None, :]
                qk_dot.masked_fill_(memory_mask, mask_value)
        
        attn = stable_softmax(qk_dot, dim=-1)
        attn = self.attn_drop(attn)
        o = einops.einsum(attn, v, 'b h i j, b h j d -> b h i d')
        o = einops.rearrange(o, 'b h n d -> b n (h d)')

        o = self.out_proj(o)
        o = self.out_drop(o)
        return o 


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, qkv_dim=None, n_head=8, ffn_hidden=1024, dropout=0.0, cross_attn = False, use_ffn=True) -> None:
        super().__init__()
        
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, qkv_dim, n_head, dropout, dropout)
        # self.attn = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        if cross_attn:
            self.norm_mem = nn.LayerNorm(d_model)
        
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, ffn_hidden)),
                ("gelu", nn.GELU()),
                ("c_proj", nn.Linear(ffn_hidden, d_model))
            ]))
            self.norm_out = nn.LayerNorm(d_model)



    def forward(
            self,
            x: torch.Tensor,
            memory: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
    ):
        # attn_bias [B, H, N, N]
        q = self.norm_q(x)
        memory = self.norm_mem(memory) if hasattr(
            self, "norm_mem") and memory is not None else None

        attn_mask = attn_mask.bool().to(q.device) if attn_mask is not None else None
        key_padding_mask = key_padding_mask.bool().to(q.device) if key_padding_mask is not None else None
        
        x = x + self.attn(
            q=q, memory=memory, mask=attn_mask, memory_mask=key_padding_mask, attn_bias=attn_bias
        )    
       
        if self.use_ffn:
            x = x + self.ffn(self.norm_out(x))
        return x



    

