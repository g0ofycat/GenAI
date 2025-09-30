import numpy as np
from src.classes.class_helper_functions import HelperFunctions
from src.configurations.class_LM_config import config

class Sampling:
    # ======== DATA ========

    sampling_strategy = config['sampling']['sampling_strategy']
    temperature = config['sampling']['temperature']
    top_k = config['sampling']['top_k']
    top_p = config['sampling']['top_p']

    # ======== MAIN ========

    @classmethod
    def _apply_sampling(cls, logits: np.ndarray) -> np.ndarray:
        dispatch = {
            "temperature": lambda logits: cls._apply_temperature_scaling(logits),
            "top_k": lambda logits: cls._apply_top_k_filtering(logits, cls.top_k),
            "top_p": lambda logits: cls._apply_top_p_filtering(logits, cls.top_p),
        }

        if cls.sampling_strategy not in dispatch:
            print(f"{cls.sampling_strategy} isn't a valid Sampling Method")
            return logits

        return dispatch[cls.sampling_strategy](logits)
        
    @classmethod
    def _apply_temperature_scaling(cls, logits: np.ndarray) -> np.ndarray:
        if cls.temperature <= 0:
            return logits
        return logits / cls.temperature
    
    @classmethod
    def _apply_top_k_filtering(cls, logits: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or k >= len(logits):
            return logits
        
        top_k_indices = np.argpartition(logits, -k)[-k:]
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[top_k_indices] = logits[top_k_indices]
        return filtered_logits
    
    @classmethod
    def _apply_top_p_filtering(cls, logits: np.ndarray, p: float) -> np.ndarray:
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