from src.classes.class_tokenizer import Tokenizer
from src.classes.class_transformer import Transformer
from src.configurations.class_LM_config import config

tokenizer = Tokenizer()
transformer = Transformer()

class Inference:
    chat_history = []

    _SPECIAL_TOKENS = {
        "[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]",
        "<|endoftext|>", "<|startoftext|>", "<|system|>", "<|user|>", "<|assistant|>",
        "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
        "[CODE]", "[/CODE]", "[PYTHON]", "[/PYTHON]", "[JAVASCRIPT]", "[/JAVASCRIPT]",
        "[HTML]", "[/HTML]", "[MATH]", "[/MATH]", "[THINKING]", "[/THINKING]",
        "<|tool_call|>", "<|tool_response|>", "[FUNCTION]", "[/FUNCTION]",
        "[JSON]", "[/JSON]", "[SUMMARY]", "[/SUMMARY]", "[CONTEXT]", "[/CONTEXT]"
    }

    @classmethod
    def Chat(cls, user_input: str, max_tokens: int = config['generation']['base_new_tokens']) -> str:
        max_history = config['chatbot']['MAX_PROMPT_HISTORY']
        if len(cls.chat_history) > max_history:
            cls.chat_history = cls.chat_history[-max_history:]

        prompt = cls._build_prompt(user_input)
        full_output = cls._generate(prompt, max_tokens)
        response = cls._extract_bot_response(full_output)

        cls.chat_history.append({"user": user_input, "bot": response})
        return response

    @classmethod
    def Infer(cls, text: str, max_tokens: int = config['generation']['base_new_tokens']) -> str:
        return cls._generate(text, max_tokens)

    @classmethod
    def _generate(cls, prompt: str, max_tokens: int) -> str:
        result_ids = transformer.generate_response(prompt, max_tokens)
        decoded = tokenizer.decode(result_ids)

        if isinstance(decoded, list):
            decoded = ''.join(map(str, decoded))

        for token in cls._SPECIAL_TOKENS:
            decoded = decoded.replace(token, '')

        return decoded.replace('\u0120', ' ').strip()

    @classmethod
    def _build_prompt(cls, user_input: str) -> str:
        system_prompt = config['chatbot']['system_prompt']
        history = "\n".join(f"User: {turn['user']}\nBot: {turn['bot']}" for turn in cls.chat_history)
        return f"{system_prompt}\n\n{history}\nUser: {user_input}\nBot: "

    @classmethod
    def _extract_bot_response(cls, full_output: str) -> str:
        last_user = f"User: {cls.chat_history[-1]['user']}" if cls.chat_history else "User:"
        parts = full_output.split(last_user)

        if len(parts) < 2:
            return full_output.strip()

        after_user = parts[-1]
        if "Bot:" not in after_user:
            return full_output.strip()

        response = after_user.split("Bot:")[-1]

        for stop_word in ("User:", "Bot:"):
            if stop_word in response:
                response = response.split(stop_word)[0]

        return response.strip()

    @classmethod
    def reset(cls):
        cls.chat_history = []