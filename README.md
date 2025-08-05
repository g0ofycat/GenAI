# Transformer Architecture (AI)

A Full transformer Architecture made in Python (Including Encoding & Decoding). A side project to sharpen my skills in ML / AL; Most likely going to optimize & organize this entire project in the future and possibly train it.

## Architecture & Design Choices

- **[GPT-2's Merge & Vocabulary JSON Files](https://huggingface.co/openai-community/gpt2/tree/main)**
- **Current default Parameter Count (When Trained):**: *~45.6 M*
- **Customizable Transformer Settings**
- **Full class for Chatbot-like inferencing and cleaning up Foreign characters**
- **Different kinds of Sampling Strategies & Tokenization Methods**
- **GeLU for the Activation Function for both the FFN's**

**Example Inferencing:**

```lua
from path.to.class_inference import Inference 

print(Inference.Chat("This is the input", 5)) # (Prompt: string, OutputtedTokens: int)
```