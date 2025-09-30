import numpy as np
from src.classes.class_tokenizer import Tokenizer
from src.classes.class_helper_functions import HelperFunctions
from src.classes.class_encode_layer import EncoderLayer
from src.classes.class_decode_layer import DecodeLayer
from src.classes.class_sampling import Sampling
from src.configurations.class_LM_config import config

class Transformer:
    def __init__(self):
        self.d_model = config['model']['d_model']
        self.num_layers = config['model']['num_layers']
        self.vocab_size = config['model']['vocab_size']

        self.max_token_input = config['chatbot']['max_token_input']

        self.tokenizer = Tokenizer()

        self.encoder_layers = [EncoderLayer() for _ in range(self.num_layers)]
        self.decoder_layers = [DecodeLayer() for _ in range(self.num_layers)]

        self.embedding = np.random.randn(self.vocab_size, self.d_model) * np.sqrt(2 / self.vocab_size)
        self.output_layer = np.random.randn(self.d_model, self.vocab_size) * np.sqrt(2 / self.d_model)

    # ======== MAIN ========

    def encode_input(self, sentence: str) -> tuple[list[int], np.ndarray]: # VECTOR MATRIX (R x C): d_model x tokens
        token_ids = self.tokenizer.encode(sentence)
        
        assert len(token_ids) <= self.max_token_input, "[encode_input]: Token input has exceeded the limit"

        if any(tok is None for tok in token_ids):
            raise ValueError(f"[FATAL] token_ids contains None: {token_ids}")

        vec_matrix = self.embedding[token_ids]

        pos_enc = HelperFunctions.PositionalEncoding(len(token_ids), self.d_model)
        
        vec_matrix = vec_matrix + pos_enc

        for layer in self.encoder_layers:
            vec_matrix = layer.forward(vec_matrix)
        
        return token_ids, vec_matrix

    def generate_response(self, prompt: str, max_new_tokens: int) -> list[int]:
        token_ids, encoder_output = self.encode_input(prompt)
        
        decoder_input = self.embedding[token_ids]
        pos_enc = HelperFunctions.PositionalEncoding(len(token_ids), self.d_model)
        x = decoder_input + pos_enc

        kv_caches = [None] * len(self.decoder_layers)
        
        for layer_idx, layer in enumerate(self.decoder_layers):
            x, kv_caches[layer_idx] = layer.forward(x, encoder_output=encoder_output, use_cache=True)
        
        for _ in range(max_new_tokens):
            last_token_vector = x[-1:]
            
            logits = last_token_vector @ self.output_layer
            logits = Sampling._apply_sampling(logits[0])
            probs = HelperFunctions.Softmax(logits)
            
            next_token = np.random.choice(len(probs), p=probs)
            token_ids.append(next_token)
            
            stop_token_id = self.tokenizer.vocab.get(config['generation']['stop_token'], None)
            if next_token == stop_token_id:
                break

            new_token_embedding = self.embedding[next_token:next_token+1]
            new_pos_enc = HelperFunctions.PositionalEncoding(len(token_ids), self.d_model)[-1:]
            x = new_token_embedding + new_pos_enc
            
            for layer_idx, layer in enumerate(self.decoder_layers):
                x, kv_caches[layer_idx] = layer.forward(
                    x, 
                    encoder_output=encoder_output, 
                    kv_cache=kv_caches[layer_idx],
                    use_cache=True
                )
        
        return token_ids