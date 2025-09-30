import math
import numpy as np

class HelperFunctions:
    # ======== ACTIVATION FUNCTIONS ========

    @staticmethod
    def Softmax(inputs: list[float]) -> list[float]:
        if isinstance(inputs, np.ndarray):
            max_value = np.max(inputs)
            exp_values = np.exp(inputs - max_value)
            return exp_values / np.sum(exp_values)
        else:
            max_value = max(inputs)
            exp_values = [math.exp(x - max_value) for x in inputs]
            exp_sum = sum(exp_values)
            return [x / exp_sum for x in exp_values]

    @staticmethod
    def GeLU(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    # ======== ATTENTION MECHANISM ========

    @staticmethod
    def Self_Attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask = None) -> np.ndarray:
        d_k = Q.shape[-1]

        scores = np.dot(Q, K.T) / math.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        weights = HelperFunctions.Softmax(scores)

        output = np.dot(weights, V)

        return output
    
    @staticmethod
    def MultiHeadAttention(x: np.ndarray, num_heads: int, d_model: int, mask = None, Q: np.ndarray = None, K: np.ndarray = None, V: np.ndarray = None) -> np.ndarray:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads
        
        if Q is None:
            seq_len = x.shape[0]
            W_q = np.random.randn(d_model, d_model)
            W_k = np.random.randn(d_model, d_model)
            W_v = np.random.randn(d_model, d_model)
            W_o = np.random.randn(d_model, d_model)
            Q = x @ W_q
            K = x @ W_k
            V = x @ W_v
        else:
            W_o = np.random.randn(d_model, d_model)
        
        seq_len = Q.shape[0]
        total_seq_len = K.shape[0]

        def split_heads(tensor: np.ndarray, seq_len: int) -> np.ndarray:
            return tensor.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

        Q_heads = split_heads(Q, seq_len)
        K_heads = split_heads(K, total_seq_len)
        V_heads = split_heads(V, total_seq_len)

        heads = []
        for i in range(num_heads):
            attn = HelperFunctions.Self_Attention(Q_heads[i], K_heads[i], V_heads[i], mask)
            heads.append(attn)

        concatenated = np.concatenate(heads, axis=-1)
        output = concatenated @ W_o

        return output
    
    @staticmethod
    def MultiHeadCrossAttention(x_q: np.ndarray, x_kv: np.ndarray, num_heads: int, d_model: int, mask = None) -> np.ndarray:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads

        seq_len_q = x_q.shape[0]
        seq_len_kv = x_kv.shape[0]

        W_q = np.random.randn(d_model, d_model)
        W_k = np.random.randn(d_model, d_model)
        W_v = np.random.randn(d_model, d_model)
        W_o = np.random.randn(d_model, d_model)

        Q = x_q @ W_q
        K = x_kv @ W_k
        V = x_kv @ W_v

        def split_heads(tensor: np.ndarray, seq_len: int) -> np.ndarray:
            return tensor.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

        Q_heads = split_heads(Q, seq_len_q)
        K_heads = split_heads(K, seq_len_kv)
        V_heads = split_heads(V, seq_len_kv)

        heads = []
        for i in range(num_heads):
            attn = HelperFunctions.Self_Attention(Q_heads[i], K_heads[i], V_heads[i], mask)
            heads.append(attn)

        concatenated = np.concatenate(heads, axis=-1)

        output = concatenated @ W_o

        return output

    # ======== MISC ========

    @staticmethod
    def CrossEntropyLoss(actual: np.ndarray, predicted: np.ndarray) -> float:
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        return -np.sum(actual * np.log(predicted + epsilon))

    @staticmethod
    def PositionalEncoding(seq_len: int, d_model: int) -> np.ndarray:
        pos_enc = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        return pos_enc
    
    @staticmethod
    def LayerNorm(x: np.ndarray, d_model: int, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(variance + epsilon)

        out = gamma * x_norm + beta

        return out