from class_tokenizer import Tokenizer
from class_transformer import Transformer
from configurations.class_LM_config import config

tokenizer = Tokenizer()
transformer = Transformer()

class Inference:
    chat_history = []

    @staticmethod
    def Chat(user_input: str, max_tokens: int = config['generation']['base_new_tokens']) -> str:
        if len(Inference.chat_history) > config['chatbot']['MAX_PROMPT_HISTORY']:
            Inference.chat_history = Inference.chat_history[-config['chatbot']['MAX_PROMPT_HISTORY']:]

        prompt = Inference._build_prompt(user_input)

        full_output = Inference._generate(prompt, max_tokens)
        response = Inference._extract_bot_response(full_output)

        Inference.chat_history.append({"user": user_input, "bot": response})

        return response

    @staticmethod
    def Infer(sentence: str, max_tokens: int = config['generation']['base_new_tokens']) -> str:
        return Inference._generate(sentence, max_tokens)

    @staticmethod
    def _generate(prompt: str, max_tokens: int) -> str:
        result_ids = transformer.generate_response(prompt, max_tokens)
        
        decoded = tokenizer.decode(result_ids)

        if isinstance(decoded, list):
            decoded = ''.join(str(token) for token in decoded)

        special_tokens = [
            "[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]",
            "<|endoftext|>", "<|startoftext|>", "<|system|>", "<|user|>", "<|assistant|>",
            "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
            "[CODE]", "[/CODE]", "[PYTHON]", "[/PYTHON]", "[JAVASCRIPT]", "[/JAVASCRIPT]",
            "[HTML]", "[/HTML]", "[MATH]", "[/MATH]", "[THINKING]", "[/THINKING]",
            "<|tool_call|>", "<|tool_response|>", "[FUNCTION]", "[/FUNCTION]",
            "[JSON]", "[/JSON]", "[SUMMARY]", "[/SUMMARY]", "[CONTEXT]", "[/CONTEXT]"
        ]

        for token in special_tokens:
            decoded = decoded.replace(token, '')

        if '\u0120' in decoded:
            decoded = decoded.replace('\u0120', ' ')

        return decoded.strip()

    @staticmethod
    def _build_prompt(user_input: str) -> str:
        prompt = config['chatbot']['system_prompt'] + "\n\n"
        
        for turn in Inference.chat_history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n"
            
        return prompt + f"User: {user_input}\nBot: "

    @staticmethod
    def _extract_bot_response(full_output: str) -> str:
        last_turn = f"User: {Inference.chat_history[-1]['user']}" if Inference.chat_history else "User:"
        split_output = full_output.split(last_turn)
        
        if len(split_output) < 2:
            return full_output.strip()

        after_last_user = split_output[-1]
        if "Bot:" not in after_last_user:
            return full_output.strip()

        response = after_last_user.split("Bot:")[-1]

        for stop_word in ["User:", "Bot:"]:
            if stop_word in response:
                response = response.split(stop_word)[0]

        return response.strip()


    @staticmethod
    def reset():
        Inference.chat_history = []