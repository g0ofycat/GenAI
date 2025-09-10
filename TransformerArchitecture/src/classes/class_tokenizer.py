import json
import os
import re
from src.configurations.class_LM_config import config

class Tokenizer:
    def __init__(self):

        VocabFile = config["tokenizer"]["vocab_path"]
 
        if VocabFile.endswith('.json'):
            with open(VocabFile, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            with open(VocabFile, "r", encoding="utf-8") as f:
                tokens = f.read().splitlines()
                self.vocab = {token: idx for idx, token in enumerate(tokens)}

        merges_path = config["tokenizer"]["merges_path"]
        
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                start_idx = 1 if lines and lines[0].startswith('#version') else 0
                self.merges = [tuple(line.strip().split()) for line in lines[start_idx:] 
                              if line and not line.startswith("#")]
        else:
            self.merges = []

        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.lowercase = config["tokenizer"]["lowercase"]
        self.tokenizer_type = config["tokenizer"]["tokenizer_type"].lower()
        self.bpe_seperator = config["tokenizer"]["bpe_seperator"]

        if "[UNK]" not in self.vocab:
            raise ValueError("Vocabulary must contain '[UNK]' token for handling unknown tokens")

        valid_types = {"whitespace", "char", "bpe"}

        if self.tokenizer_type not in valid_types:
            raise ValueError(f"Invalid Tokenizer Type: {self.tokenizer_type}")
        
    def _merge_pair(self, tokens: list, pair) -> list[str]:
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]: # "If the current index of the token we're on is the same as the pair as well as the index after then append both"
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def _get_pairs(self, tokens: list) -> set[tuple]:
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs
        
    def _split_on_punctuation(self, text: str) -> list[str]:
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def ApplyBPE(self, text: str) -> list[str]:
        words = self._split_on_punctuation(text)
        final_tokens = []

        for i, word in enumerate(words):
            if word.isalnum():
                tokens = list(word)

                while True:
                    pairs = self._get_pairs(tokens)
                    candidate = None

                    for merge_pair in self.merges:
                        if merge_pair in pairs:
                            candidate = merge_pair
                            break 

                    if candidate is None:
                        break

                    tokens = self._merge_pair(tokens, candidate)
  
                if i > 0 and len(tokens) > 0:
                    tokens[0] = self.bpe_seperator + tokens[0]
                    
            else:
                tokens = [word]

            final_tokens.extend(tokens)

        return final_tokens
    
    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        if self.tokenizer_type == "whitespace":
            return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        elif self.tokenizer_type == "char":
            return list(text)
        elif self.tokenizer_type == "bpe":
            return self.ApplyBPE(text)
        
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        tokenized_text = self.tokenize(text)
        
        tokens = []
        
        for token in tokenized_text:
            if token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                tokens.append(self.vocab["[UNK]"])
        
        return tokens

    def decode(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        
        decoded_tokens = []
        
        for token_id in token_ids:
            if not isinstance(token_id, int):
                try:
                    token_id = int(token_id)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid token_id type: {type(token_id)}, skipping")
                    continue

            if token_id < 0 or token_id >= len(self.id_to_token):
                print(f"Warning: token_id {token_id} out of vocabulary range, using [UNK]")
                decoded_tokens.append("[UNK]")
            elif token_id in self.id_to_token:
                decoded_tokens.append(self.id_to_token[token_id])
            else:
                print(f"Warning: Unknown token_id {token_id}, using [UNK]")
                decoded_tokens.append("[UNK]")
        
        return decoded_tokens