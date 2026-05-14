from dataclass import dataclass
import torch
import torch.nn as nn
# from torch.nn import functional as F
# from torch.utils.data import Dataset, DataLoader

@dataclass
class GPTConfig:
    vocab_size: int = 65  # Size of the vocabulary
    block_size: int = 256    # Maximum context length
    n_layer: int = 6         # Number of transformer layers
    n_head: int = 6           # Number of attention heads
    n_embd: int = 384          # Dimensionality of embeddings
    # dropout: float = 0.1       # Dropout rate

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.Transformer(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
       