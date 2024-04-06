from .loader import GPT2Loader

class Model:
    def __init__(self, name: str, model_dir: str) -> None:
        loader = GPT2Loader(model_dir)
        self.tokenizer = loader._load_tokenizer()
    
    def gen(self, prompt: str) -> str:
        # tokenize
        tokens =  self.tokenizer.encode(prompt)
        
        return self.tokenizer.decode(tokens)