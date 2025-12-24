# self-attention：Q、K、V 来自同一个张量

# cross-attention的维度：
# Q 来自 decoder
# K、V 来自 encoder
# T_q ≠ T_k = T_v（长度可以不同）
# K 和 V 必须来自同一个 source
import torch.nn as nn
import math
import torch


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        # Linear会自动把最后一个维度当作特征。
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        n_batch, seq_len, d_input = x.shape

        qkv = self.qkv(x)
        # 拆分qkv：3*[Batch, Token, d_model]
        q, k, v = qkv.chunk(3, dim=-1)
        # 下面拆分多头并为矩阵乘法做前处理
        # [Batch, Token, dim]变成[Batch, Token, n_head, d_head]变成[Batch, n_head, Token, d_head]
        q = q.view(n_batch, seq_len, -1, self.d_head).transpose(1, 2)
        k = k.view(n_batch, seq_len, -1, self.d_head).transpose(1, 2)
        v = v.view(n_batch, seq_len, -1, self.d_head).transpose(1, 2)
        # 做每个 batch、每个 head做矩阵乘法：[Token, d_head] @ [d_head, Token]得到[Batch, n_head, Token_q, Token_k]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # softmax不改变维度
        if mask is not None:
            # mask 应该是 0/1 或 False/True
            # 构建下三角矩阵的mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            # 升维
            mask = mask.unsqueeze(0).unsqueeze(0)
            # mask矩阵[1,1,T,T] 可广播到 [B, n_head, T, T]
            # mask == 0 是个判断，表示mask中false的位置被替换为-inf。
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        # [Batch, n_head, Token_q, Token_k] @ [Batch, n_head, Token_v, d_head]等于[Batch, n_head, Token_q, d_head]
        out = (attn @ v)
        # [Batch, n_head, Token_q, d_head] 变成 [Batch, Token_q, n_head, d_head] 再合并多头得到[Batch, Token_q, d_model]
        out = out.transpose(1, 2).contiguous().view(n_batch, seq_len, d_input)

        return self.out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MHA(d_model, n_heads)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # ffn设计可以用CNN。看下面3点：
        # 1.相邻token是否强相关？
        # 2模式是否平移不变？
        # 3是否算力 / 数据有限？
        # ≥2个“是” → CNN ≥2个“否” → MLP
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # Self-Attention + Residual + LN
        x = self.ln1(x + self.attn(x))
        # FFN + Residual + LN
        x = self.ln2(x + self.ffn(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return x

# tensor=torch.randn(2,10,512)
# encoder=TransformerEncoder(num_layers=4,d_model=512,n_heads=8,d_ff=2048, out_dim=1)
# output=encoder(tensor)
# print(output)
