import GPT2
import GPT2.model

# GPT2.GPT2Loader("model")._load_tokenizer()
model = GPT2.model.Model("gpt2", "model")
print(model.gen("The transformer architecture used in Large Language Models is"))
