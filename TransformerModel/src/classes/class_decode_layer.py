import numpy as np
from typing import Tuple, Optional
from src.classes.class_helper_functions import HelperFunctions
from src.configurations.class_LM_config import config

class DecodeLayer:
    def __init__(self):
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['attention_heads']
        self.d_ff = 4 * self.d_model

        self.gamma1 = np.ones(self.d_model)
        self.beta1 = np.zeros(self.d_model)

        self.gamma2 = np.ones(self.d_model)
        self.beta2 = np.zeros(self.d_model)

        self.gamma3 = np.ones(self.d_model)
        self.beta3 = np.zeros(self.d_model)
        
        self.W1 = np.random.randn(self.d_model, self.d_ff) * np.sqrt(2 / self.d_model)
        self.b1 = np.zeros(self.d_ff)

        self.W2 = np.random.randn(self.d_ff, self.d_model) * np.sqrt(2 / self.d_ff)
        self.b2 = np.zeros(self.d_model)

        self.W_q_self = np.random.randn(self.d_model, self.d_model) * np.sqrt(2 / self.d_model)
        self.W_k_self = np.random.randn(self.d_model, self.d_model) * np.sqrt(2 / self.d_model)
        self.W_v_self = np.random.randn(self.d_model, self.d_model) * np.sqrt(2 / self.d_model)
        self.W_o_self = np.random.randn(self.d_model, self.d_model) * np.sqrt(2 / self.d_model)

    # ======== HELPERS ========

    def generate_causal_mask(self, seq_len: int) -> np.ndarray:
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
  
    def feedforward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = HelperFunctions.GeLU(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2

    # ======== MAIN ========

    def forward(self, x: np.ndarray, encoder_output: Optional[np.ndarray] = None, kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None, use_cache: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        seq_len = x.shape[0]

        Q = x @ self.W_q_self
        
        if kv_cache is not None:
            cached_K, cached_V = kv_cache
            K_new = x @ self.W_k_self
            V_new = x @ self.W_v_self
            K = np.concatenate([cached_K, K_new], axis=0)
            V = np.concatenate([cached_V, V_new], axis=0)
        else:
            K = x @ self.W_k_self
            V = x @ self.W_v_self
        
        total_seq_len = K.shape[0]
        
        if seq_len == 1 and kv_cache is not None:
            causal_mask = None
        else:
            causal_mask = self.generate_causal_mask(seq_len) if kv_cache is None else \
                        np.triu(np.ones((seq_len, total_seq_len), dtype=bool), k=total_seq_len - seq_len + 1)
        
        self_attn_out = HelperFunctions.MultiHeadAttention(x, self.num_heads, self.d_model, mask=causal_mask, Q=Q, K=K, V=V)
        x = HelperFunctions.LayerNorm(x + self_attn_out, self.d_model, self.gamma1, self.beta1)
        
        if encoder_output is not None:
            cross_attn_out = HelperFunctions.MultiHeadCrossAttention(x, encoder_output, self.num_heads, self.d_model)
            x = HelperFunctions.LayerNorm(x + cross_attn_out, self.d_model, self.gamma2, self.beta2)
        
        ffn_out = self.feedforward(x)
        x = HelperFunctions.LayerNorm(x + ffn_out, self.d_model, self.gamma3, self.beta3)
        
        new_cache = (K, V) if use_cache else None
        return x, new_cache