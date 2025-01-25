#!/usr/bin/env python
# coding: utf-8

# In[1]:


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# In[2]:


import torch
import torch.nn as nn
from torch.nn import functional as F

# Set random seed for reproducibility across all devices
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
torch.manual_seed(1337)


# In[3]:


# Import required PyTorch modules for neural network operations
import torch
import torch.nn as nn
from torch.nn import functional as F

# Function to set random seed for reproducibility
def set_seed(seed):
    # Set seed for CPU operations
    torch.manual_seed(seed)
    # Set seed for GPU operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Set the random seed to 1337
torch.manual_seed(1337)

# Define the LSTM-based language model class
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, device='cuda'):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Store the device (CPU/GPU) for model operations
        self.device = device
        
        # Create embedding layer to convert token indices to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim).to(device)
        # Create LSTM layer with specified parameters
        self.lstm = nn.LSTM(
            input_size=embedding_dim,        # Size of input features (embedding dimension)
            hidden_size=hidden_dim,          # Number of LSTM hidden units
            num_layers=num_layers,           # Number of stacked LSTM layers
            dropout=0.1 if num_layers > 1 else 0,  # Apply dropout between LSTM layers
            batch_first=True                 # Expect input shape as (batch, sequence, features)
        ).to(device)
        # Layer normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.1)
        # Output layer to convert LSTM outputs to vocabulary logits
        self.output_layer = nn.Linear(hidden_dim, vocab_size).to(device)

    def forward(self, idx, targets=None):
        # Move input tensor to specified device
        idx = idx.to(self.device)
        # Move targets to device if provided
        if targets is not None:
            targets = targets.to(self.device)

        # Get batch size and sequence length
        B, T = idx.shape
        # Convert token indices to embeddings
        token_emb = self.token_embedding_table(idx)
        
        # Pass embeddings through LSTM (ignore hidden states)
        lstm_out, _ = self.lstm(token_emb)
        # Apply layer normalization to LSTM outputs
        x = self.layer_norm(lstm_out)
        # Apply dropout for regularization
        x = self.dropout(x)
        # Generate logits for each token in vocabulary
        logits = self.output_layer(x)

        # If no targets provided, skip loss calculation
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for loss calculation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Calculate cross entropy loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Move input tensor to specified device
        idx = idx.to(self.device)
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Truncate sequence if too long (for memory efficiency)
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            # Get model predictions
            logits, _ = self(idx_cond)
            # Take logits of last token only
            logits = logits[:, -1, :]
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append new token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# In[4]:


class TextDataHandler:
    def __init__(self, text, device='cuda'):
        self.device = device
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])
        
        # Encode data and move to device
        self.data = torch.tensor(self.encode(text), dtype=torch.long).to(device)

    def get_batch(self, batch_size, block_size):
        ix = torch.randint(len(self.data) - block_size, (batch_size,), device=self.device)
        x = torch.stack([self.data[i:i+block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+block_size+1] for i in ix])
        return x, y

class ModelTrainer:
    def __init__(self, model, data_handler, config):
        self.model = model
        self.data_handler = data_handler
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['max_iters']
        )

    def train(self):
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        
        for iter in range(self.config['max_iters']):
            # Get batch
            xb, yb = self.data_handler.get_batch(
                self.config['batch_size'], 
                self.config['block_size']
            )

            # Training step
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            logits, loss = self.model(xb, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            self.scheduler.step()

            # Logging
            if iter % self.config['eval_interval'] == 0:
                print(f"iter {iter}: loss {loss.item():.4f}")


    def generate_sample(self, prompt=None, max_new_tokens=100, temperature=1.0):
        """
        Generate text sample based on optional prompt.
        Args:
            prompt (str, optional): Input text to continue from. If None, starts with empty context
            max_new_tokens (int): Number of tokens to generate
            temperature (float): Higher values (>1.0) increase randomness, lower values (<1.0) make text more deterministic
        """
        print("\nGenerating sample text:")
        self.model.eval()  # Set to evaluation mode

        with torch.no_grad():
            # Handle the context based on prompt
            if prompt is None:
                # If no prompt, start with empty context
                context = torch.zeros((1, 1), dtype=torch.long, device=self.config['device'])
            else:
                # Encode the prompt text to token indices
                encoded_prompt = self.data_handler.encode(prompt)
                context = torch.tensor([encoded_prompt], dtype=torch.long, device=self.config['device'])

            # Generate the completion
            generated_ids = self.model.generate(context, max_new_tokens)[0].tolist()

            # If we used a prompt, print it along with the generation
            if prompt is not None:
                print(f"Prompt: {prompt}")
                print("Generated completion:")
                # Only decode the new tokens (excluding the prompt tokens)
                generated_text = self.data_handler.decode(generated_ids[len(encoded_prompt):])
                print(prompt + generated_text)
            else:
                # If no prompt, decode and print the entire generation
                generated_text = self.data_handler.decode(generated_ids)
                print(generated_text)

        self.model.train()  # Set back to training mode
        print('='*40)

        return generated_text


# In[6]:


# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set configurations
config = {
    'device': device,
    'batch_size': 64,
    'block_size': 256,
    'max_iters': 15000,
    'eval_interval': 500,
    'sample_interval': 500,
    'learning_rate': 3e-4,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
}



# Initialize data handler
data_handler = TextDataHandler(text, device=device)

# Initialize model
model = LSTMLanguageModel(
    vocab_size=data_handler.vocab_size,
    embedding_dim=384,
    hidden_dim=512,
    device=device
)



# In[7]:


trainer = ModelTrainer(model, data_handler, config)


# In[ ]:


# Train model
trainer.train()


# In[ ]:





# In[ ]:




