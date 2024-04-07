from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('./model')  # or any other checkpoint
word_embeddings = model.transformer.wte.weight  # Word Token Embeddings 
print(word_embeddings[0])
# position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings 