# LSTM Text Generator 🚀

A PyTorch implementation of a character-level LSTM language model for creative text generation. This model can learn patterns from any text corpus and generate new text in similar style.

## 🌟 Features

- Character-level text generation using LSTM
- Customizable model architecture (embedding size, hidden dimensions, layers)
- Temperature-controlled text generation
- CUDA support for faster training
- Interactive text generation with custom prompts

## 🔧 Installation

```bash
git clone https://github.com/yourusername/lstm-text-generator.git
cd lstm-text-generator
pip install -r requirements.txt
```

## 💻 Usage

### Training

```python
from src.model import LSTMLanguageModel
from src.data_handler import TextDataHandler
from src.trainer import ModelTrainer

# Load your text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize components
data_handler = TextDataHandler(text)
model = LSTMLanguageModel(
    vocab_size=data_handler.vocab_size,
    embedding_dim=384,
    hidden_dim=512
)

# Configure training
config = {
    'batch_size': 64,
    'block_size': 256,
    'max_iters': 15000,
    'learning_rate': 3e-4,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'eval_interval': 500
}

# Train
trainer = ModelTrainer(model, data_handler, config)
trainer.train()

# Save model
torch.save(model.state_dict(), 'model.pth')
```

### Generating Text

```python
# Load saved model
model = LSTMLanguageModel(vocab_size=data_handler.vocab_size)
model.load_state_dict(torch.load('model.pth'))

# Generate text
generated_text = trainer.generate_sample(
    prompt="Once upon a time",
    max_new_tokens=200,
    temperature=0.8
)
print(generated_text)
```

## 🔬 Model Architecture

- Embedding Layer: Converts character indices to dense vectors
- Multi-layer LSTM: Processes sequential data
- Layer Normalization: Stabilizes training
- Dropout: Prevents overfitting
- Linear Output Layer: Projects to vocabulary space

## 📈 Training Tips

- Use longer sequences (block_size) for better context understanding
- Adjust temperature during generation (lower for focused text, higher for creativity)
- Increase embedding_dim and hidden_dim for more complex patterns
- Use gradient clipping to prevent exploding gradients

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

## 📝 License

MIT License
