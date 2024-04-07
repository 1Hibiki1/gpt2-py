import GPT2
import GPT2.model

model = GPT2.model.Model("gpt2", "model")
print(model.gen("The transformer architecture used in Large Language Models is"))
