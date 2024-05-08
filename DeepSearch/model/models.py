import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from DeepSearch.model.modules import *
from DeepSearch.utils.tokenizer import Tokenizer


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            context_dim: int = 512,
            n_head: int = 8,
            n_queries: int = 256,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = Attention(d_model, None, n_head,
                              k_dim=context_dim, v_dim=context_dim)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(context_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.norm_k(x)

        batch_size = x.shape[0]
        query = repeat(self.query, 'n d -> b n d', b=batch_size)
        query = self.norm_q(query)
        out = self.attn(query, x, memory_mask=mask)
        return out


class MLP(nn.Module):
    def __init__(self, d_model, ffn_hidden, out_dim) -> None:
        super().__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.out_dim = out_dim
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(ffn_hidden, out_dim)

    def forward(self, bias):
        # bias: [B, N, N, D]
        bias = self.act(self.linear1(bias))
        bias = self.linear2(bias)
        return bias


class BiasProj(nn.Module):
    def __init__(self, d_model, out_dim) -> None:
        super().__init__()
        self.out_proj = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, bias):
        # bias: [B, N, N, D]
        bias_head = self.out_proj(bias)
        bias_head = rearrange(bias_head, 'b i j d -> b d i j')
        return bias_head


class PeakEncoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 qkv_dim=None,
                 output_dim=512,
                 n_head=8,
                 n_layer=12,
                 ffn_hidden=1024,
                 dropout=0.,
                 n_queries=1) -> None:
        super().__init__()
        self.d_model = d_model
        self.qkv_dim = qkv_dim if qkv_dim is not None else d_model
        self.output_dim = output_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ffn = ffn_hidden
        self.dropout = dropout
        self.n_queries = n_queries

        
        self.peak_emd = nn.Sequential(
            nn.Linear(1, d_model//2, bias=True),
            nn.GELU('tanh'),
            nn.Linear(d_model//2, d_model, bias=True))
        
        self.pos_emd = NaiveMZPositionalEmbedding(d_model)
        self.encoder_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=d_model,
                n_head=n_head,
                ffn_hidden=ffn_hidden,
                dropout=dropout)
            for _ in range(n_layer)
        ])

        self.attn_pooler = AttentionalPooler(output_dim, d_model, n_queries=1)
        self.norm_final = nn.LayerNorm(output_dim)
        self.proj = nn.Linear(output_dim, output_dim, bias=False)

        # self.init_parameters()

    def init_parameters(self):
        proj_std = (self.d_model ** -0.5) * ((2 * self.n_layer) ** -0.5)
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5
        for block in self.encoder_blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.ffn.c_fc.weight, std=fc_std)
            nn.init.normal_(block.ffn.c_proj.weight, std=proj_std)

        nn.init.normal_(self.proj.weight, std=self.d_model ** -0.5)

    
    def forward(self, spectra, mask):
        """forward 

        Args:
            spectra (Tensor): spectra with shape [B, N, 2] (first dim m/z)
            mask (Tensor): mask with dim of shape [B, N]
        """
        # emd peaks and add postional emd
        dtype = spectra.dtype

        x = self.peak_emd(spectra[:, :, [1]]) + \
            self.pos_emd(spectra).to(dtype)

        for layer in self.encoder_blocks:
            x = layer(x, key_padding_mask=mask)
        pooled = self.attn_pooler(x, mask)
        pooled = self.norm_final(pooled)
        pooled = self.proj(pooled).squeeze(1)

        return pooled, x


class UniModalDecoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 output_dim=512,
                 n_head=8,
                 ffn_hiden=1024,
                 n_layer=12,
                 dropout=0.,
                 pad_id=0,
                 seq_len=34,
                 vocab_size=25,
                 n_cls=1,
                 bias_dim=None,
                 use_attn_bias=False) -> None:
        
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pad_id = pad_id
        self.n_cls = n_cls
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.cls_emd = nn.Parameter(torch.empty(d_model))
        self.positional_emd = PositionalEmbedding(d_model, seq_len + n_cls)
        self.n_head = n_head
        self.n_layer = n_layer
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(d_model=d_model,
                                    n_head=n_head,
                                    ffn_hidden=ffn_hiden,
                                    dropout=dropout)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.register_buffer(
            'attn_mask', self.build_attention_mask(), persistent=False)
        self.cls_norm = nn.LayerNorm(d_model)
        self.seq_proj = nn.Linear(d_model, output_dim, bias=False)
        self.output_dim = output_dim

        if use_attn_bias:
            assert bias_dim is not None

            self.in_bias_mlp = nn.Sequential(
                nn.Linear(bias_dim, d_model // 4, bias=False),
                MLP(d_model // 4, d_model // 2, out_dim=d_model//4))
        
            self.bias_layers = nn.ModuleList([
                BiasProj(d_model // 4, n_head)
                for _ in range(n_layer)
            ])

        # self.init_parameters()

    def build_attention_mask(self):
        mask = torch.empty(self.seq_len + self.n_cls,
                           self.seq_len + self.n_cls)
        mask.fill_(1.)
        mask.tril_(0)

        return mask

    def build_cls_mask(self, pep: torch.Tensor):
        dtype = self.seq_proj.weight.dtype
        cls_mask = (pep != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (0, 1, cls_mask.shape[2], 0), value=1.0)
        return cls_mask

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_emd, std=0.01)

        proj_std = (self.d_model ** -0.5) * \
            ((2 * self.n_layer) ** -0.5)
        attn_std = self.d_model ** -0.5

        fc_std = (2 * self.d_model) ** -0.5
        for block in self.layers:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.ffn.c_fc.weight, std=fc_std)
            nn.init.normal_(block.ffn.c_proj.weight, std=proj_std)

        nn.init.normal_(self.seq_proj.weight,
                        std=self.d_model ** -0.5)

    def forward(self, pep, use_mask=True, attn_bias=None):
        # attn_bias [B, N, N, d_attn_bias]
        seq_len = pep.shape[1] + 1
        x = self.token_embedding(pep)  # [batch_size, N, d_model]

        cls_emd = repeat(self.cls_emd, 'd -> b 1 d', b=x.shape[0])
        x = torch.cat([x, cls_emd], dim=1)
        attn_mask = None
        bias_head = None
        if use_mask:
            attn_mask = self.attn_mask
            cls_mask = self.build_cls_mask(pep)

            attn_mask = attn_mask[None, :seq_len,
                                  :seq_len] * cls_mask[:, :seq_len, :seq_len]
            attn_mask = ~(attn_mask.bool())  # [B, N, N]

        x = self.positional_emd(x)
        if attn_bias is not None and hasattr(self, 'in_bias_mlp'):
            attn_bias = self.in_bias_mlp(attn_bias)

        for i_layer in range(self.n_layer):
            layer = self.layers[i_layer]
            if attn_bias is not None and hasattr(self, 'bias_layers'):
                bias_layer = self.bias_layers[i_layer]
                bias_head = bias_layer(attn_bias)

            x = layer(x, attn_mask=attn_mask, attn_bias=bias_head)

        pooled, tokens = x[:, -self.n_cls], x[:, :-self.n_cls]
        pooled = self.cls_norm(pooled)
        pooled = self.seq_proj(pooled)

        return pooled, tokens, attn_bias


class MultiModalDecoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 output_dim=512,
                 n_head=8,
                 ffn_hidden=1024,
                 n_layer=12,
                 dropout=0.,
                 pad_id=0,
                 seq_len=34,
                 use_attn_bias=False,
                 bias_dim=12) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.ffn_hidden = ffn_hidden
        self.n_layer = n_layer
        self.dropout = dropout
        self.pad_id = pad_id
        self.seq_len = seq_len
        self.bias_dim = bias_dim
        if use_attn_bias:
            self.bias_layers = nn.ModuleList([
                BiasProj(d_model//4, n_head)
                for _ in range(n_layer)
            ])

        self.layers = nn.ModuleList([
            ResidualAttentionBlock(d_model=d_model,
                                   n_head=n_head,
                                   ffn_hidden=ffn_hidden,
                                   dropout=dropout,
                                   use_ffn=False)
            for _ in range(n_layer)
        ])
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(d_model=d_model,
                                   n_head=n_head,
                                   ffn_hidden=ffn_hidden,
                                   dropout=dropout, cross_attn=True)
            for _ in range(n_layer)
        ])

        self.register_buffer(
            'attn_mask', self.build_attention_mask(), persistent=False)

        self.norm = nn.LayerNorm(d_model)
        self.seq_proj = nn.Linear(d_model, output_dim, bias=False)
        # self.init_parameters()

    def init_parameters(self):
        proj_std = (self.d_model ** -0.5) * \
            ((2 * self.n_layer) ** -0.5)
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5
        for block in self.layers:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        for block in self.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.ffn.c_fc.weight, std=fc_std)
            nn.init.normal_(block.ffn.c_proj.weight, std=proj_std)

        nn.init.normal_(self.seq_proj.weight, std=self.d_model ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        mask = torch.empty(self.seq_len, self.seq_len)
        mask.fill_(1.)
        mask.triu_(1)  # zero out the lower diagonal
        return mask.bool()

    def forward(self, spectra_emds, spectra_mask, seq_emds, attn_bias=None):
        B, seq_len = seq_emds.shape[0], seq_emds.shape[1]
        # device= seq_emds.device

        bias_head = None
        for i_layer in range(self.n_layer):
            block = self.layers[i_layer]
            cross_attn = self.cross_attn[i_layer]
            if attn_bias is not None and hasattr(self, 'bias_layers'):
                bias_layer = self.bias_layers[i_layer]
                bias_head = bias_layer(attn_bias)
                bias_head = bias_head[:, :, :seq_len, :seq_len]

            seq_emds = block(
                seq_emds, attn_mask=self.attn_mask[None, :seq_len, :seq_len], attn_bias=bias_head)
            seq_emds = cross_attn(
                seq_emds, memory=spectra_emds, key_padding_mask=spectra_mask)

        seq_emds = self.norm(seq_emds)

        seq_emds = self.seq_proj(seq_emds)

        return seq_emds


class DeepSearch(nn.Module):
    def __init__(self, 
                 spectrum_dim=1024,
                 spectrum_hidden=3072,
                 n_enc_layer=10,
                 peptide_dim=1024,
                 peptide_hidden=3072,
                 n_uni_dec_layer=10,
                 n_mul_dec_layer=10,
                 output_dim=512, 
                 n_head=12, 
                 dropout=0., 
                 pad_id=0, 
                 bias_dim=36, 
                 vocab_size=25,
                 use_bias=True) -> None:
        super().__init__()
        self.tokenizer = Tokenizer()
        
        self.peak_encoder = PeakEncoder(
            d_model=spectrum_dim,
            output_dim=output_dim, 
            n_head=n_head, 
            n_layer=n_enc_layer, 
            ffn_hidden=spectrum_hidden,
            dropout=dropout,
            
        )

        self.unimodal_decoder = UniModalDecoder(
            d_model=peptide_dim, 
            output_dim=output_dim, 
            n_head=n_head, 
            ffn_hiden=peptide_hidden, 
            n_layer=n_uni_dec_layer, 
            dropout=dropout, 
            pad_id=pad_id,
            vocab_size=vocab_size,
            use_attn_bias=use_bias, 
            bias_dim=bias_dim,

        )
        
        self.multimodal_decoder = MultiModalDecoder(
            d_model = peptide_dim, 
            output_dim=output_dim, 
            n_head=n_head, 
            ffn_hidden=peptide_hidden, 
            n_layer=n_mul_dec_layer, 
            dropout=dropout, 
            use_attn_bias=use_bias,
            bias_dim=bias_dim)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id


    def _encode_spectra(self, spectra, mask, normalize=True):
        spectra_latent, spectra_token = self.peak_encoder(spectra, mask)
        spectra_latent = F.normalize(spectra_latent, dim=-1) if normalize else spectra_latent

        return spectra_latent, spectra_token

    def _encode_peptide(self, peptide, normalize=True, use_mask=True, attn_bias=None):
        pep_latent, pep_token, attn_bias = self.unimodal_decoder(peptide, use_mask, attn_bias=attn_bias)
        pep_latent = F.normalize(pep_latent, dim=-1) if normalize else pep_latent
        return pep_latent, pep_token, attn_bias
    

    def forward(self, spectra, spectra_mask, peptide, return_pep_emd=False, attn_bias=None):
        batch_size = peptide.shape[0]
        device = peptide.device
        peptide, labels = peptide[:, :-1], peptide[:, 1:]
        
        pep_latent, pep_token, attn_bias = self._encode_peptide(peptide, attn_bias=attn_bias)
        if return_pep_emd:
            return pep_latent

        spectra_latent, spectra_token = self._encode_spectra(spectra, spectra_mask)
        logits = self.multimodal_decoder(spectra_token, spectra_mask, pep_token, attn_bias=attn_bias)

        return {
            "logits": logits,
            "pep_latents": pep_latent,
            "spectra_latents": spectra_latent,
            "labels": labels,
            "temperature": self.temperature.exp()}

