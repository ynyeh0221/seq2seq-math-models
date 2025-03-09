# Mathematical Operations with Transformer Models

This repository contains implementations of transformer-based models for basic arithmetic operations (addition, subtraction, multiplication, and division) and a transfer learning approach that combines knowledge from these individual models.

## Overview

The project demonstrates how transformer models can learn to perform arithmetic operations when trained on appropriate datasets. Each basic operation (addition, subtraction, multiplication, and division) has its own dedicated model, and a fifth model utilizes transfer learning to combine knowledge from all four operations.

## Requirements

```
torch >= 1.12.0
numpy >= 1.21.0
```

To install the required packages:
```bash
pip install torch numpy
```

## Model Architecture

All models use a transformer-based architecture with:
- Embedding layer to convert tokens to vectors
- Positional encoding to maintain sequence order information
- Transformer encoder-decoder architecture
- Linear output layer for token prediction

Key components include:
- **Tokenizer**: Converts mathematical expressions to token sequences
- **Dataset**: Handles data generation and preparation
- **PositionalEncoding**: Adds positional information to embeddings
- **TransformerModel**: The core model architecture

## Individual Operation Models

### Addition Model

The addition model learns to predict the sum of two numbers.

- Input format: `a+b=`
- Output format: `c` (where `c = a + b`)
- Examples: `12+34=` → `46`

```python
# Example usage
model = TransformerModel(vocab_size=len(tokenizer.vocab), embed_dim=128, num_heads=4)
model.load_state_dict(torch.load('models/addition_model.pth'))
# Predict: 25+37=
```

### Subtraction Model

The subtraction model learns to predict the difference between two numbers.

- Input format: `a-b=`
- Output format: `c` (where `c = a - b`)
- Examples: `43-21=` → `22`

### Multiplication Model

The multiplication model learns to predict the product of two numbers.

- Input format: `a*b=`
- Output format: `c` (where `c = a * b`)
- Examples: `6*7=` → `42`

### Division Model

The division model learns to predict the quotient of two numbers.

- Input format: `a/b=`
- Output format: `c` (where `c = a / b`, rounded to specified decimal places)
- Examples: `10/2=` → `5`

## Migration Learning Model

The migration learning model combines knowledge from all four operation models to create a single model that can handle any of the basic arithmetic operations.

- Input format: `a+b=`, `a-b=`, `axb=`, or `a/b=` (using 'x' for multiplication)
- Output format: The result of the specified operation
- Training approach: Initializes with averaged weights from pre-trained individual models and fine-tunes on a combined dataset with all operations

```python
# Example usage of the migration learning model
model = TransformerModel(vocab_size=len(tokenizer.vocab), embed_dim=128, num_heads=4, num_layers=4)
model.load_state_dict(torch.load('models/migration_model.pth'))
# Can predict any of: 25+37=, 43-21=, 6x7=, 10/2=
```

## Training Process

Each model is trained on a dataset of randomly generated examples for its specific operation:

1. Generate examples with inputs and expected outputs
2. Encode the data using the tokenizer
3. Train the transformer model for a specified number of epochs
4. Save the trained model weights

The training process uses:
- CrossEntropyLoss for token prediction
- Adam optimizer
- Learning rate of 0.0001
- Gradient clipping to prevent exploding gradients

## Evaluation

Models are evaluated on test sets of unseen examples. The evaluation metrics include:
- Accuracy: Percentage of completely correct answers
- Partial match: For longer outputs, the percentage of correctly predicted digits

Example evaluation output:
```
Input: 57+28= Predicted: 85 True: 85
Input: 64+39= Predicted: 103 True: 103
```

## Migration Learning Approach

The migration learning script:
1. Loads the pre-trained weights from each of the four operation models (addition, subtraction, multiplication, and division)
2. Creates a new model with a shared architecture and increased number of layers (4 instead of 3)
3. Calculates the average weights from all four models for each parameter
4. Initializes the new model with these averaged weights
5. Fine-tunes on a combined dataset containing all four operations
6. Saves the resulting model that can handle multiple operations

This averaging approach allows the model to begin training with a good initialization that incorporates knowledge from all four individual models, making the learning process more efficient and potentially improving overall performance.

## Usage Examples

```python
# Load a model
model = TransformerModel(vocab_size=len(tokenizer.vocab), embed_dim=128, num_heads=4).to(device)
model.load_state_dict(torch.load('models/addition_model.pth'))

# Make predictions
def predict(model, expression, tokenizer, device, max_length=10):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([tokenizer.encode(expression)], dtype=torch.long).to(device)
        trg_input = torch.tensor([[tokenizer.vocab2idx[SOS]]], device=device)
        
        for _ in range(max_length):
            trg_mask = model.transformer.generate_square_subsequent_mask(trg_input.size(1)).to(device)
            output = model(src, trg_input, trg_mask)
            pred_token = output[:, -1, :].argmax(dim=-1)
            
            if pred_token.item() == tokenizer.vocab2idx[EOS]:
                break
                
            trg_input = torch.cat([trg_input, pred_token.unsqueeze(0)], dim=1)
        
        pred_seq = tokenizer.decode(trg_input[0][1:])
        return ''.join(pred_seq)

# Example
result = predict(model, "45+67=", tokenizer, device)
print(f"45+67= {result}")  # Should output: 45+67= 112
```

## Future Improvements

Potential enhancements for this project include:
- Support for more complex expressions with multiple operations
- Handling of decimal numbers and fractions
- Increased model capacity for more complex calculations
- Exploration of different model architectures
- Web interface for easy demonstration
