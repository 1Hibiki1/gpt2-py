from .loader import GPT2Loader

class Model:
    def __init__(self, name: str, model_dir: str) -> None:
        loader = GPT2Loader(model_dir)
        self.tokenizer = loader.load_tokenizer()
        loader._load_tensors()

    def prepare(self, prompt: str) -> None:
        #prep kv cache
        
        pass
    
    def gen(self, prompt: str) -> str:
        # tokenize
        tokens =  self.tokenizer.encode(prompt)


        
        return self.tokenizer.decode(tokens)