import torch.nn as nn
import torch

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab              # token -> id
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        # 最简单：按空格切
        tokens = text.lower().split()
        ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]
        return ids

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "i": 2,
    "love": 3,
    "you": 4
}

tokenizer = SimpleTokenizer(vocab)
# tokenizer是分词器，把原始文本切分成 token，并把每个 token 映射成字典中的整数 id
# 最终返回的是一个整数序列。
input_ids = tokenizer.encode("i love you")

class Embeddings(nn.Module):
    def __init__(self, vocab_size, max_len, d_model):
        super().__init__()
        # 把tokenizer返回的整数序列编码成d_model维度的向量
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # 把位置信息编码成d_model维度的向量
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        return self.token_emb(input_ids) + self.pos_emb(pos_ids)

