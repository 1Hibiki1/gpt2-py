import json

import numpy as np
import regex

type Embedding = np.array
type Token = str
type Tensor = np.array


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
        tensot_np_array = np.frombuffer(tensor_data_buf, dtype="f")
        tensot_np_array = tensot_np_array.reshape(tuple(tensor_meta["shape"]))

        return tensot_np_array


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


class GPT2KVCache:
    def __init__(self) -> None:
        self.k_cache = list()
        self.v_cache = list()

        self.cur_idx = 0

    def append_kv(self, kv_tuple) -> None:
        self.k_cache.append(kv_tuple[0])
        self.v_cache.append(kv_tuple[1])

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        l = len(self.k_cache)
        if self.cur_idx >= l:
            raise StopIteration
        kv = (self.k_cache[self.cur_idx], self.v_cache[self.cur_idx])
        self.cur_idx += 1
        return kv


class GPT2Layer:
    def __init__(self) -> None:
        self.q_weights: np.array = None
        self.q_bias: np.array = None

        self.k_weights: np.array = None
        self.k_bias: np.array = None

        self.v_weights: np.array = None
        self.v_bias: np.array = None

        self.kv_cache = GPT2KVCache()

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

    def forward(self, input: Embedding) -> Embedding:
        # 1) norm
        norm = (input - np.mean(input)) / np.std(input)
        norm = norm * self.ln_1_weights + self.ln_1_bias

        # 2) compute query?
        query = np.dot(norm, self.q_weights) + self.q_bias

        # split attention head stuf
        query = query.reshape((12, 64))

        output = list()
        # 3) for all previous tokens, compute softmax(QK.t/sqrt(d))*V and append res to output
        for k, v in self.kv_cache:
            kq = query * k
            kq = np.sum(kq, axis=1)
            kq = kq / np.sqrt(768 / 12)
            # TODO: mask kq? ugh what????
            kq = self._softmax(kq)

            v_scaled = (v.T * kq).T

            final_v = v_scaled.reshape(768)

            output.append(final_v)

        output = np.array(output)
        # 4) proj
        output = np.dot(output, self.attn_proj_weights) + self.attn_proj_bias

        # residual stufff
        output += input

        # 5) norm 2
        norm = (output - np.mean(output)) / np.std(output)
        norm = norm * self.ln_2_weights + self.ln_2_bias

        # 6) mlp
        fc_out = np.dot(norm, self.mlp_fc_weights) + self.mlp_fc_bias
        fc_out = self._gelu(fc_out)

        proj_out = np.dot(fc_out, self.mlp_proj_weights) + self.mlp_proj_bias
        proj_out = proj_out + output

        # 7) "sum" all the outputs, and thats the input to the next layer (apparently)
        return np.sum(proj_out, axis=0)

    def cache_kv(self, input: Embedding) -> None:
        k = np.dot(input, self.k_weights) + self.k_bias
        v = np.dot(input, self.v_weights) + self.v_bias

        # TODO: dont hardcode the shape lol
        self.kv_cache.append_kv((k.reshape(12, 64), v.reshape(12, 64)))


class GPT2:
    def __init__(self, dir: str = "model") -> None:
        self.tokenizer = Tokenizer(dir)
        self.token_embedding_list = list()

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

            cur_layer.q_weights, cur_layer.k_weights, cur_layer.v_weights = np.split(
                qkv_weights, 3, 1
            )
            cur_layer.q_bias, cur_layer.k_bias, cur_layer.v_bias = np.split(
                qkv_bias, 3, 0
            )

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

    def prepare(self, prompt: str) -> None:
        token_list = self.tokenizer.tokenize(prompt)
        code_list = self.tokenizer.encode(token_list)

        # get embeddings
        self.token_embedding_list = np.take(self.wte, code_list, axis=0)

        # wte + wpe
        position_embeddings = np.take(self.wpe, range(len(code_list)), axis=0)
        self.token_embedding_list += position_embeddings

        # KV cache for each token for each layer
        for token_emb in self.token_embedding_list:
            for layer in self.layers:
                layer.cache_kv(token_emb)

    def get_next_token(self) -> str:
        # get last item in self.token_embedding_list
        cur_embedding = self.token_embedding_list[-1]

        # do the stuf
        for layer in self.layers:
            cur_embedding = layer.forward(cur_embedding)
            layer.cache_kv(cur_embedding)

        # final norm
        cur_embedding = (cur_embedding - np.mean(cur_embedding)) / np.std(cur_embedding)
        cur_embedding = cur_embedding * self.ln_f_weights + self.ln_f_bias

        # mul with wte and get arg max
        logits = np.dot(cur_embedding, self.wte.T)
        output_token = np.argmax(logits)

        # append embedding for new token
        new_embed = self.wte[output_token]
        new_embed = new_embed + self.wpe[len(self.token_embedding_list)]
        self.token_embedding_list = np.append(
            self.token_embedding_list, [new_embed], axis=0
        )

        # return token
        return self.tokenizer.decode_single(output_token)


# always raises some overflow error, idk why, this whole implementation is probs real messed up
# np.seterr(all='raise')

model = GPT2("model")
prompt = "The transformer architecture used in large language models is"
model.prepare(prompt)
print(prompt, end="", flush=True)
for _ in range(10):
    print(model.get_next_token(), end="", flush=True)
print()
