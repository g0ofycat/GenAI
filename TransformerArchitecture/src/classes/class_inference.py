from src.classes.class_tokenizer import Tokenizer
from src.classes.class_transformer import Transformer
from src.configurations.class_LM_config import config
from typing import List, Dict, Optional
import threading

class Inference:
    _chat_history: List[Dict[str, str]] = []
    _lock = threading.Lock()
    _tokenizer = None
    _transformer = None

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = Tokenizer()
            
        return cls._tokenizer

    @classmethod
    def _get_transformer(cls):
        if cls._transformer is None:
            cls._transformer = Transformer()

        return cls._transformer

    @classmethod
    def chat_history(cls) -> List[Dict[str, str]]:
        with cls._lock:
            return cls._chat_history.copy()

    @classmethod
    def Chat(cls, user_input: str, max_tokens: Optional[int] = None) -> str:
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty or whitespace only")

        max_tokens = max_tokens or config['generation']['base_new_tokens']

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        with cls._lock:
            max_history = config['chatbot']['MAX_PROMPT_HISTORY']

            if len(cls._chat_history) > max_history:
                cls._chat_history = cls._chat_history[-max_history:]

            prompt = cls._build_prompt(user_input)
            
        try:
            full_output = cls._generate(prompt, max_tokens)
            response = cls._extract_bot_response(full_output, user_input)
        except Exception as e:
            raise RuntimeError(f"Chat generation failed: {e}")

        with cls._lock:
            cls._chat_history.append({"user": user_input, "bot": response})
        
        return response

    @classmethod
    def Infer(cls, text: str, max_tokens: Optional[int] = None) -> str:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only")
        
        max_tokens = max_tokens or config['generation']['base_new_tokens']

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        try:
            return cls._generate(text, max_tokens)
        except Exception as e:
            raise RuntimeError(f"Inference generation failed: {e}")

    @classmethod
    def _generate(cls, prompt: str, max_tokens: int) -> str:
        try:
            transformer = cls._get_transformer()
            tokenizer = cls._get_tokenizer()
            
            result_ids = transformer.generate_response(prompt, max_tokens)
            decoded = tokenizer.decode(result_ids)

            if isinstance(decoded, list):
                decoded = ''.join(map(str, decoded))

            return decoded.replace(config['tokenizer']['bpe_seperator'], ' ').strip()
        
        except Exception as e:
            raise RuntimeError(f"Generation process failed: {e}")

    @classmethod
    def _build_prompt(cls, user_input: str) -> str:
        try:
            system_prompt = config['chatbot']['system_prompt']

            history = "\n".join(f"User: {turn['user']}\nBot: {turn['bot']}" for turn in cls._chat_history)

            return f"{system_prompt}\n\n{history}\nUser: {user_input}\nBot: "
        except KeyError as e:
            raise RuntimeError(f"Missing configuration key: {e}")

    @classmethod
    def _extract_bot_response(cls, full_output: str, current_user_input: str) -> str:
        try:
            if not cls._chat_history:
                if "Bot:" in full_output:
                    response = full_output.split("Bot:")[-1]
                    if "User:" in response:
                        response = response.split("User:")[0]
                    return response.strip()
                return full_output.strip()

            last_user_prompt = f"User: {current_user_input}"
            parts = full_output.split(last_user_prompt)

            if len(parts) < 2:
                if "Bot:" in full_output:
                    response = full_output.split("Bot:")[-1]
                    if "User:" in response:
                        response = response.split("User:")[0]
                    return response.strip()
                return full_output.strip()

            after_user = parts[-1]
            if "Bot:" not in after_user:
                return full_output.strip()

            response = after_user.split("Bot:")[-1]

            for stop_word in ("User:", "Bot:"):
                if stop_word in response:
                    response = response.split(stop_word)[0]

            return response.strip()
        
        except Exception:
            return full_output.strip()

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._chat_history = []

    @classmethod
    def get_history_length(cls) -> int:
        with cls._lock:
            return len(cls._chat_history)

    @classmethod
    def get_last_exchange(cls) -> Optional[Dict[str, str]]:
        with cls._lock:
            return cls._chat_history[-1].copy() if cls._chat_history else None