import numpy as np
from class_helper_functions import HelperFunctions
from configurations.class_LM_config import config

class EncoderLayer:
    def __init__(self):
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['attention_heads']
        self.d_ff = 4 * self.d_model

        self.gamma1 = np.ones(self.d_model)
        self.beta1 = np.zeros(self.d_model)
        self.gamma2 = np.ones(self.d_model)
        self.beta2 = np.zeros(self.d_model)
        
        self.W1 = np.random.randn(self.d_model, self.d_ff) * 0.01
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, self.d_model) * 0.01
        self.b2 = np.zeros(self.d_model)

    def feedforward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = HelperFunctions.GeLU(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2

    def forward(self, x: np.ndarray):
        attn_out = HelperFunctions.MultiHeadAttention(x, self.num_heads, self.d_model, mask = None)

        x = HelperFunctions.LayerNorm(x + attn_out, self.d_model, self.gamma1, self.beta1)

        ffn_out = self.feedforward(x)
        
        x = HelperFunctions.LayerNorm(x + ffn_out, self.d_model, self.gamma2, self.beta2)
        
        return x