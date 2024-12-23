# This script shows how to run LLM from local .safetensoers file
# HOW TO RUN
# download the following files into ./Llama-3.2-3B/original folder:
#  - https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/model-00001-of-00002.safetensors
#  - https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/model-00002-of-00002.safetensors
#  - https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/original/consolidated.00.pth
# then run:
# pipenv shell
# python llama3.py

from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
import torch

model_dir = './Llama-3.2-3B/original/'
device = 0 if torch.cuda.is_available() else -1

print("Loading model...")
model = LlamaForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  
    device_map="auto"  
)
print("Model loaded.")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("Tokenizer loaded.")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Pad token set to eos_token.")
else:
    print("Pad token not defined.")

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

while True:
    prompt = input("\n>> User: ")

    generated = text_generator(
        prompt,
        max_length=250,            
        min_length=50,             
        num_return_sequences=1,    
        no_repeat_ngram_size=2,    
        repetition_penalty=1.2,    
        temperature=0.7,           
        top_p=0.9,                 
        top_k=50,                  
        truncation=True
    )
    print("\n>> Bot: ")
    print(generated[0]['generated_text'])
