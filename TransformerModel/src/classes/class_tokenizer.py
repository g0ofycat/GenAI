import json
import os
import re
from typing import Dict, List, Set, Tuple
from src.configurations.class_LM_config import config

class Tokenizer:
    def __init__(self):
        self.vocab = self._load_vocab(config["tokenizer"]["vocab_path"])
        self.merges = self._load_merges(config["tokenizer"]["merges_path"])
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.lowercase = config["tokenizer"]["lowercase"]
        self.tokenizer_type = config["tokenizer"]["tokenizer_type"].lower()
        self.bpe_separator = config["tokenizer"]["bpe_seperator"]

        self._validate()

    # ======== LOADING ========

    def _load_vocab(self, path: str) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                return json.load(f)
            return {tok: idx for idx, tok in enumerate(f.read().splitlines())}

    def _load_merges(self, path: str) -> List[Tuple[str, str]]:
        if not os.path.exists(path):
            return []
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        start = 1 if lines and lines[0].startswith("#version") else 0
        return [tuple(line.split()) for line in lines[start:] if line and not line.startswith("#")]

    # ======== VALIDATION ========

    def _validate(self):
        if "[UNK]" not in self.vocab:
            raise ValueError("Vocabulary must contain '[UNK]'")
        if self.tokenizer_type not in {"whitespace", "char", "bpe"}:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    # ======== HELPERS ========

    def _merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        merged_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def _get_pairs(self, tokens: List[str]) -> Set[Tuple[str, str]]:
        return { (tokens[i], tokens[i+1]) for i in range(len(tokens)-1) }

    def _split_on_punctuation(self, text: str) -> List[str]:
        return re.findall(r'\w+|[^\w\s]', text)

    # ======== TOKENIZATION ========

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()

        method = getattr(self, f"_tokenize_{self.tokenizer_type}")
        return method(text)

    def _tokenize_whitespace(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def _tokenize_char(self, text: str) -> List[str]:
        return list(text)

    def _tokenize_bpe(self, text: str) -> List[str]:
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
                    tokens[0] = self.bpe_separator + tokens[0]
                    
            else:
                tokens = [word]

            final_tokens.extend(tokens)

        return final_tokens

    # ======== ENCODING / DECODING ========

    def encode(self, text: str) -> List[int]:
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

    def decode(self, token_ids: List[int]) -> List[str]:
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