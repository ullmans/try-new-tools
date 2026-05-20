from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import random
import time

@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Size of the vocabulary
    block_size: int = 1024    # Maximum context length
    n_layer: int = 12         # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768          # Dimensionality of embeddings
    bias: bool = True         # Whether to use bias in linear layers
    # dropout: float = 0.1       # Dropout rate


class MyDataset(Dataset):
    def __init__(self, file_path, seq_length=256, tokenizer=None):
        """
        Custom dataset for language modeling.
        
        Args:
            file_path: Path to the text file
            seq_length: Sequence length T (context length for each sample)
            tokenizer: tiktoken tokenizer. If None, uses gpt2 encoding.
        """
        # Load text
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        else:
            self.tokenizer = tokenizer
        
        # Tokenize the entire text
        self.tokens = self.tokenizer.encode(self.text)
        self.seq_length = seq_length
        
        print(f"Dataset loaded: {len(self.text)} characters, {len(self.tokens)} tokens")
    
    def __len__(self):
        """Return number of possible sequences."""
        return max(0, len(self.tokens) - self.seq_length)
    
    def __getitem__(self, idx):
        """
        Get a sequence and its target.
        x: tokens at positions [idx, idx+seq_length)
        y: tokens at positions [idx+1, idx+seq_length+1) (shifted by 1)
        
        This ensures y[i] is the expected next token after x[:i+1].
        """
        # Input sequence
        x = torch.tensor(self.tokens[idx:idx + self.seq_length], dtype=torch.long)
        
        # Target sequence (shifted by 1)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        
        return x, y


def create_dataloader(file_path, batch_size=32, seq_length=256, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the custom dataset.
    
    Args:
        file_path: Path to the text file
        batch_size: Batch size B
        seq_length: Sequence length T
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
    
    Returns:
        dataloader: PyTorch DataLoader
        tokenizer: tiktoken tokenizer for encoding/decoding
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = MyDataset(file_path, seq_length=seq_length, tokenizer=tokenizer)
    dataloader = MyDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, tokenizer


class MyDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        Custom DataLoader that yields batches of (x, y) pairs.
        Args:
            dataset: An instance of MyDataset
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data at the start of each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            batch_x = []
            batch_y = []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            yield torch.stack(batch_x), torch.stack(batch_y)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MyMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        assert self.head_dim * self.n_head == self.n_embd, "Embedding dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj.NANOGPT_SCALE_INIT = 1  # Mark for special initialization

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
        self.fc2.NANOGPT_SCALE_INIT = 1  # Mark for special initialization
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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight shering sceme
        self.lm_head.weight = self.transformer.wte.weight
        # init weights
        self.apply(self._init_weights)

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
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


if __name__ == "__main__":
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    if device == 'cuda':
        torch.cuda.manual_seed_all(1337)
        # Enable TF32 for faster computation on supported GPUs
        torch.set_float32_matmul_precision('high')
    # ============== CREATE DATALOADER ==============
    print("\n" + "="*50)
    print("Creating Shakespeare DataLoader...")
    print("="*50)
    
    # Create dataloader with your dataset
    dataloader, tokenizer = create_dataloader(
        file_path='shakespeare_dataset.txt',
        batch_size=16,      # B = 16 samples per batch
        seq_length=1024,     # T = 256 tokens per sequence
        shuffle=True
    )
    
    # Initialize model config
    config = GPTConfig()
    
    # ============== INITIALIZE MODEL ==============
    print("\n" + "="*50)
    print("Initializing GPT Model...")
    print("="*50)
    model = GPT(config).to(device)
    model = torch.compile(model)  # Optional: compile the model for faster execution (PyTorch 2.0+)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ============== TRAINING SETUP ==============
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # ============== TRAINING LOOP ==============
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            t0 = time.time()
            # Move to device
            x = x.to(device)      # Shape: (B, T)
            y = y.to(device)      # Shape: (B, T)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Forward pass
                logits = model(x)     # Shape: (B, T, vocab_size)   
                # Compute loss
                # Reshape for cross entropy: (B*T, vocab_size) and (B*T,)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            t1 = time.time()
            dt = (t1 - t0) * 1000  # Time in milliseconds
            total_loss += loss.item()
            num_batches += 1
            
            # print loss every 10 batches
            # if (batch_idx + 1) % 10 == 0:
            token_per_sec  = dataloader.batch_size * dataloader.dataset.seq_length / (dt / 1000)
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, Time: {dt:.4f}ms, tok/sec: {token_per_sec:.2f}")

            # generate text every 250 batches
            if (batch_idx + 1) % 250 == 0:
                prompt = "Hello, I'm a language model,"
                prompt_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
                                
                # Generate 100 more tokens
                with torch.no_grad():
                    for _ in range(100):
                        if input_ids.shape[1] > config.block_size:
                            # Keep only the last block_size tokens
                            input_ids = input_ids[:, -config.block_size:]
                        
                        logits = model(input_ids)
                        next_logits = logits[0, -1, :]
                        
                        # Temperature-based sampling
                        next_logits = next_logits / 0.8  # temperature = 0.8
                        probs = torch.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Decode and print
                generated_tokens = input_ids[0].tolist()
                generated_text = tokenizer.decode(generated_tokens)
                print(f"{generated_text}'\n")
               
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")
    
    # ============== EXAMPLE: SAMPLE FROM MODEL ==============
    print("\n" + "="*50)
    print("Generating Text...") 
    print("="*50)
    
    model.eval()
    
    # Start with a prompt
    # prompt = "The quality of mercy"
    # prompt_tokens = tokenizer.encode(prompt)
    # input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    # print(f"\nPrompt: '{prompt}'\n")
    
    # # Generate 100 more tokens
    # with torch.no_grad():
    #     for _ in range(100):
    #         if input_ids.shape[1] > config.block_size:
    #             # Keep only the last block_size tokens
    #             input_ids = input_ids[:, -config.block_size:]
            
    #         logits = model(input_ids)
    #         next_logits = logits[0, -1, :]
            
    #         # Temperature-based sampling
    #         next_logits = next_logits / 0.8  # temperature = 0.8
    #         probs = torch.softmax(next_logits, dim=-1)
    #         next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
    #         input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # # Decode and print
    # generated_tokens = input_ids[0].tolist()
    # generated_text = tokenizer.decode(generated_tokens)
    # print(f"Generated: '{generated_text}'\n")
    
    print("Training complete!")
x