import regex

class GPT2Tokenizer:
    def __init__(self, vocab, merges) -> None:
        self.vocab = vocab
        self.merges = merges
        self.re = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def encode(self, prompt: str):
        # FIXME: this tokenizer is very sus
    
        # TODO: this clumps consecutive spaces into a single string, so those
        # will not be found in the vocab. Need to split them into individual spaces
        tokens = regex.findall(self.re, prompt)
        tokens = [i.replace(' ', 'Ġ') for i in tokens]

        bpe_bytes = list()
        for t in tokens:
            bpe_byte = self.vocab.get(t)
            if bpe_byte != None:
                bpe_bytes.append(bpe_byte)
    
        return bpe_bytes

    def decode(self, bpe_list):
        output_str = ""
        for e in bpe_list:
            # FIXME: Will raise StopIteration if no match is found
            key = next(key for key, value in self.vocab.items() if value == e)
            output_str += key

        output_str = output_str.replace('Ġ', ' ')
        return output_str
    
class GPT2:
    def __init__(self) -> None:
        self.wte = None
