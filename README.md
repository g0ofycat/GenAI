# Transformer Architecture (AI)

A full Transformer architecture built in Python (including both encoding and decoding). Side project.

## Architecture & Design Choices

- **[GPT-2's Merge & Vocabulary Files](https://huggingface.co/openai-community/gpt2/tree/main)**
- **Default Parameter Count (When Trained):** *~45.6 M*
- **Fully Customizable Transformer Settings**
- **Chatbot class for Inference + Foreign Character Cleanup**
- **Multiple Sampling Strategies & Tokenization Methods**
- **GeLU Activation for Both FFN Layers**

**Example Inference (prompt: string, output_tokens?: int):**

```py
from path.to.class_inference import Inference

print(Inference.Chat("This is the input", 5))
```

## TODO 

- [x] Create a full Transformer
- [ ] Revise & Use proper Weight & Bias initialization before I train it
- [ ] Polish