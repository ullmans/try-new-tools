from dataclasses import dataclass
import torch
import torch.nn as nn
# from torch.nn import functional as F
# from torch.utils.data import Dataset, DataLoader

@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Size of the vocabulary
    block_size: int = 1024    # Maximum context length
    n_layer: int = 12         # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768          # Dimensionality of embeddings
    # dropout: float = 0.1       # Dropout rate


class MyMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        assert self.head_dim * self.n_head == self.n_embd, "Embedding dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size() # (B, T, C)
        
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * n_embd)
        qkv = qkv.view(batch_size, seq_length, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_head, seq_length, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (batch_size, n_head, seq_length, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, n_head, seq_length, seq_length)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, n_head, seq_length, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, n_head, head_dim)
        attn_output = attn_output.view(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, n_embd)

        output = self.out_proj(attn_output)  # (batch_size, seq_length, n_embd)
        return output
class MyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU()
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        return x
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MyMultiheadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MyMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
       
    def forward(self, idx):
        batch_size, seq_length = idx.size()
        assert seq_length <= self.config.block_size, "Sequence length exceeds block size"

        token_embeddings = self.transformer.wte(idx)  # (batch_size, seq_length, n_embd)

        position_ids = torch.arange(seq_length, device=idx.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.transformer.wpe(position_ids)  # (batch_size, seq_length, n_embd)

        x = token_embeddings + position_embeddings  # (batch_size, seq_length, n_embd)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # (batch_size, seq_length, n_embd)
        
        logits = self.lm_head(x)  # (batch_size, seq_length, vocab_size)
        return logits

# weight tying - needs to have 'transformers' import to run this peace of code, but we will ignore it for now.
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    override_args = override_args or {} # default to empty dict
    # only dropout can be overridden see more notes below
    assert all(k == 'dropout' for k in override_args)
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    print("forcing vocab_size=50257, block_size=1024, bias=True")
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    config_args['bias'] = True # always True for GPT model checkpoints
    # we can override the dropout rate, if desired
    if 'dropout' in override_args:
        print(f"overriding dropout rate to {override_args['dropout']}")
        config_args['dropout'] = override_args['dropout']
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model

model = GPT.from_pretrained('gpt2')
print("didn't crash yay!!")