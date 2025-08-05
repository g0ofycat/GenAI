import numpy as np
from class_tokenizer import Tokenizer
from class_helper_functions import HelperFunctions
from class_encode_layer import EncoderLayer
from class_decode_layer import DecodeLayer
from configurations.class_LM_config import config

class Transformer:
    def __init__(self):
        self.d_model = config['model']['d_model']
        self.learning_rate = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.attention_heads = config['model']['attention_heads']
        self.num_layers = config['model']['num_layers']
        self.temperature = config['generation']['temperature']
        self.vocab_size = config['model']['vocab_size']
        self.max_token_input = config['chatbot']['max_token_input']
        
        self.top_k = config['generation']['top_k']
        self.top_p = config['generation']['top_p']
        
        self.sampling_strategy = config['generation']['sampling_strategy']
        
        self.tokenizer = Tokenizer()

        self.embedding = np.random.randn(self.vocab_size, self.d_model) * 0.01

        self.encoder_layers = [EncoderLayer() for _ in range(self.num_layers)]
        self.decoder_layers = [DecodeLayer() for _ in range(self.num_layers)]
        
        self.output_layer = np.random.randn(self.d_model, self.vocab_size) * 0.01

    def encode_input(self, sentence: str) -> tuple[list[int], np.ndarray]: # VECTOR MATRIX (R x C): d_model x tokens
        token_ids = self.tokenizer.encode(sentence)
        
        assert len(token_ids) <= self.max_token_input, "[encode_input]: Token input has exceeded the limit" # Includes system prompt

        if any(tok is None for tok in token_ids):
            raise ValueError(f"[FATAL] token_ids contains None: {token_ids}")

        vec_matrix = self.embedding[token_ids]

        pos_enc = HelperFunctions.PositionalEncoding(len(token_ids), self.d_model)
        
        vec_matrix = vec_matrix + pos_enc

        for layer in self.encoder_layers:
            vec_matrix = layer.forward(vec_matrix)
        
        return token_ids, vec_matrix

    def generate_response(self, prompt: str, max_new_tokens: int = config['generation']['base_new_tokens']) -> list[int]:
        token_ids, encoder_output = self.encode_input(prompt)

        for _ in range(max_new_tokens):
            decoder_input = self.embedding[token_ids]
            pos_enc = HelperFunctions.PositionalEncoding(len(token_ids), self.d_model)
            x = decoder_input + pos_enc

            for layer in self.decoder_layers:
                x = layer.forward(x, encoder_output = encoder_output)

            last_token_vector = x[-1]
            logits = last_token_vector @ self.output_layer
            logits = self._apply_sampling(logits)
            probs = HelperFunctions.Softmax(logits)

            next_token = np.random.choice(len(probs), p = probs)
            token_ids.append(next_token)

            stop_token_id = self.tokenizer.vocab.get(config['generation']['stop_token'], None)
            if next_token == stop_token_id:
                break

        return token_ids

    def _apply_sampling(self, logits: np.ndarray) -> np.ndarray:
        valid_sample_types = {"top_k", "top_p", "temperature"}
        
        if self.sampling_strategy not in valid_sample_types:
            print(f"{self.sampling_strategy} isn't a valid Sampling Method")
            return logits

        if self.sampling_strategy == "temperature":
            return self._apply_temperature_scaling(logits)
        elif self.sampling_strategy == "top_k":
            return self._apply_top_k_filtering(logits, self.top_k)
        elif self.sampling_strategy == "top_p":
            return self._apply_top_p_filtering(logits, self.top_p)
        
        return logits
        
    def _apply_temperature_scaling(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            return logits
        return logits / self.temperature
    
    def _apply_top_k_filtering(self, logits: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or k >= len(logits):
            return logits
        
        top_k_indices = np.argpartition(logits, -k)[-k:]
        
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[top_k_indices] = logits[top_k_indices]
        
        return filtered_logits
    
    def _apply_top_p_filtering(self, logits: np.ndarray, p: float) -> np.ndarray:
        if p >= 1.0:
            return logits
        
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        probs = HelperFunctions.Softmax(sorted_logits)

        cumulative_probs = np.cumsum(probs)
        
        cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
        cutoff_idx = max(1, cutoff_idx)
        
        filtered_logits = np.full_like(logits, -np.inf)
        selected_indices = sorted_indices[:cutoff_idx]
        filtered_logits[selected_indices] = logits[selected_indices]
        
        return filtered_logits