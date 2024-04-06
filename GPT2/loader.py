from .gpt2 import GPT2Tokenizer
import json

class GPT2Loader:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
    
    def _load_tokenizer(self) -> GPT2Tokenizer:
        f = open(f"{self.model_dir}/tokenizer.json")
        tok_data = json.load(f)
        # for i in tok_data["model"]:
        #     print(i)
        # print(len(tok_data["model"]["vocab"]))
        # print(len(tok_data["model"]["merges"]))
        return GPT2Tokenizer(tok_data["model"]["vocab"], tok_data["model"]["merges"])

    def _load_tensors(self):
        f = open(f"{self.model_dir}/model.safetensors")
        return