import json

import numpy as np
import regex
import tiktoken

type Embedding = np.array
type Token = str
type Tensor = np.array

def _layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params

class Loader:
    def __init__(
        self, dir: str = "model", tensor_file: str = "model.safetensors"
    ) -> None:
        self.safetensors_file = open(f"{dir}/{tensor_file}", "rb")
        self.meta_len = int.from_bytes(self.safetensors_file.read(8), "little")
        self.tensor_meta = json.loads(
            self.safetensors_file.read(self.meta_len).decode("utf-8")
        )

    def get_tensor(self, name: str) -> Tensor:
        tensor_data_start_offset = 8 + self.meta_len
        tensor_meta = self.tensor_meta[name]

        tensor_start_offset = tensor_meta["data_offsets"][0] + tensor_data_start_offset
        tensor_end_offset = tensor_meta["data_offsets"][1] + tensor_data_start_offset

        self.safetensors_file.seek(tensor_start_offset)
        tensor_data_buf = self.safetensors_file.read(
            tensor_end_offset - tensor_start_offset
        )
        tensor_np_array = np.frombuffer(tensor_data_buf, dtype="f")
        if len(tensor_meta["shape"]) == 1:
            tensor_meta["shape"] = [1] + tensor_meta["shape"]
        tensor_np_array = np.reshape(tensor_np_array, tensor_meta["shape"])

        return tensor_np_array


class Tokenizer:
    def __init__(self, dir: str = "model", tok_file: str = "tokenizer.json") -> None:
        f = open(f"{dir}/{tok_file}")
        tok_data = json.load(f)

        self.vocab = tok_data["model"]["vocab"]

        # from openai lol
        self.re = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def tokenize(self, input: str) -> list[str]:
        tokens = regex.findall(self.re, input)
        tokens = [i.replace(" ", "Ġ") for i in tokens]
        # TODO: merge tokens(?)
        return tokens

    def encode(self, tokens: list[str]) -> list[int]:
        bpe_bytes = list()
        for t in tokens:
            bpe_byte = self.vocab.get(t)
            if bpe_byte != None:
                bpe_bytes.append(bpe_byte)

        return bpe_bytes

    def decode(self, code_list: list[int]) -> list[Token]:
        output_str = ""
        for e in code_list:
            try:
                key = next(
                    key for key, value in self.vocab.items() if value == e
                )  # from some stack overflow post
                output_str += key
            except:
                # suposed to raise StopIteration if no match is found
                pass

        output_str = output_str.replace("Ġ", " ")
        return output_str

    def decode_single(self, code: int) -> Token:
        try:
            key = next(key for key, value in self.vocab.items() if value == code)
            return key.replace("Ġ", " ")
        except:
            return ""


class GPT2Layer:
    def __init__(self) -> None:
        self.attn_weights = None
        self.attn_bias = None

        self.attn_proj_weights = None
        self.attn_proj_bias = None

        self.ln_1_weights = None
        self.ln_1_bias = None

        self.ln_2_weights = None
        self.ln_2_bias = None

        self.mlp_fc_weights = None
        self.mlp_fc_bias = None
        self.mlp_proj_weights = None
        self.mlp_proj_bias = None

        self.ln_f_weights = None
        self.ln_f_bias = None

    def _softmax(self, x):
        x = x / np.linalg.norm(x)
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, input: list[Embedding]) -> list[Embedding]:
        # norm
        norm1 = _layer_norm(input, self.ln_1_weights, self.ln_1_bias)
        
        # attn (q,k,v)
        qkv = np.dot(norm1, self.attn_weights)
        qkv = np.add(qkv, self.attn_bias)

        # self attn
        q, k, v = np.split(qkv, 3, -1)
        q_split = np.split(q, 12, -1)
        k_split = np.split(k, 12, -1)
        v_split = np.split(v, 12, -1)

        mask = (1 - np.tri(len(input))) * -1e10

        out_split = list()
        for i in range(12):
            out_split.append(self._softmax(((q_split[i] @ k_split[i].T)/np.sqrt(q_split[i].shape[-1])) + mask) @ v_split[i])

        attn_out = np.hstack(out_split)

        # proj
        attn_proj = attn_out @ self.attn_proj_weights + self.attn_proj_bias

        # add input
        inp_ffn = attn_proj + input

        # FFN

        # norm 2
        ffn_norm = _layer_norm(inp_ffn, self.ln_2_weights, self.ln_2_bias)
        # mlp
        ffn_out = (ffn_norm @ self.mlp_fc_weights) + self.mlp_fc_bias
        ffn_out = self._gelu(ffn_out)
        # proj
        ffn_out = (ffn_out @ self.mlp_proj_weights) + self.mlp_proj_bias

        return (ffn_out + inp_ffn)


class GPT2:
    def __init__(self, dir: str = "model") -> None:
        self.tokenizer = Tokenizer(dir)
        self.encoder = tiktoken.get_encoding("r50k_base")

        self.loader = Loader()
        self.wte: np.array = self.loader.get_tensor("wte.weight")
        self.wpe: np.array = self.loader.get_tensor("wpe.weight")
        self.ln_f_weights = self.loader.get_tensor(f"ln_f.weight")
        self.ln_f_bias = self.loader.get_tensor(f"ln_f.bias")

        self.layers: list[GPT2Layer] = list()

        # TODO: dont hardcode num layers
        for i in range(12):
            cur_layer = GPT2Layer()

            qkv_weights = self.loader.get_tensor(f"h.{i}.attn.c_attn.weight")
            qkv_bias = self.loader.get_tensor(f"h.{i}.attn.c_attn.bias")

            cur_layer.attn_weights = qkv_weights
            cur_layer.attn_bias = qkv_bias

            cur_layer.attn_proj_weights = self.loader.get_tensor(
                f"h.{i}.attn.c_proj.weight"
            )
            cur_layer.attn_proj_bias = self.loader.get_tensor(f"h.{i}.attn.c_proj.bias")
            cur_layer.ln_1_weights = self.loader.get_tensor(f"h.{i}.ln_1.weight")
            cur_layer.ln_1_bias = self.loader.get_tensor(f"h.{i}.ln_1.bias")
            cur_layer.ln_2_weights = self.loader.get_tensor(f"h.{i}.ln_2.weight")
            cur_layer.ln_2_bias = self.loader.get_tensor(f"h.{i}.ln_2.bias")
            cur_layer.mlp_fc_weights = self.loader.get_tensor(f"h.{i}.mlp.c_fc.weight")
            cur_layer.mlp_fc_bias = self.loader.get_tensor(f"h.{i}.mlp.c_fc.bias")
            cur_layer.mlp_proj_weights = self.loader.get_tensor(
                f"h.{i}.mlp.c_proj.weight"
            )
            cur_layer.mlp_proj_bias = self.loader.get_tensor(f"h.{i}.mlp.c_proj.bias")

            self.layers.append(cur_layer)

    def get_next_token(self, prompt: str) -> str:
        # encode
        # token_list: list[int] = self.tokenizer.encode(self.tokenizer.tokenize(prompt))
        token_list = self.encoder.encode(prompt)
        # print()
        # print(token_list)

        # wte + wpe
        final_encoding = self.wte[token_list] + self.wpe[range(len(token_list))]

        # forawrd n layers
        cur_embedding = final_encoding
        for l in self.layers:
            cur_embedding = l.forward(cur_embedding)

        # final norm
        output = _layer_norm(cur_embedding, self.ln_f_weights, self.ln_f_bias)

        # wte * output
        output = np.dot(self.wte, output.T)
        output = output.T[-1]

        output_token = np.argmax(output)

        # return self.tokenizer.decode_single(output_token)
        return self.encoder.decode([output_token])

model = GPT2("model")

prompt = "Alan Turing theorized that computers would one day become"
print(prompt, end="", flush=True)

for _ in range(8):
    next_tok = model.get_next_token(prompt)
    print(next_tok, end="", flush=True)
    prompt += next_tok

print()
