
from transformers import GPTJForCausalLM, AutoTokenizer, pipeline
import torch
import pickle

#"EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
model = pipeline('text-generation', model='EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
with open('model.pkl', 'wb') as f:
   pickle.dump(model, f)