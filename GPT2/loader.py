from .gpt2 import GPT2Tokenizer, GPT2
import json
import numpy as np

class GPT2Loader:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.gpt2_model = GPT2()
    
    def load_tokenizer(self) -> GPT2Tokenizer:
        f = open(f"{self.model_dir}/tokenizer.json")
        tok_data = json.load(f)
        return GPT2Tokenizer(tok_data["model"]["vocab"], tok_data["model"]["merges"])

    def _load_tensors(self):
        print("loading tensors")
        f = open(f"{self.model_dir}/model.safetensors", "rb")

        N = int.from_bytes(f.read(8), "little")
        print(f"Header length: {N}")

        header = json.loads(f.read(N).decode("utf-8"))
        for i in header:
            print(i)

        wte_meta = header['wte.weight']
        print(wte_meta)

        wte_start_byte = wte_meta['data_offsets'][0] + 8 + N
        wte_end_byte = wte_meta['data_offsets'][1] + 8 + N

        f.seek(wte_start_byte)
        wte_buf = f.read(wte_end_byte - wte_start_byte)

        wte_np = np.frombuffer(wte_buf, dtype='f')

        wte_np = wte_np.reshape(tuple(wte_meta['shape']))
        self.gpt2_model.wte = wte_np
        # print(wte_np[0])

        return

    def load_model(self) -> GPT2:
        self._load_tensors()
        return self.gpt2_model