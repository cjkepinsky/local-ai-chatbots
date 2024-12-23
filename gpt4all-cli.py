# https://docs.gpt4all.io/gpt4all_python.html#quickstart

from gpt4all import GPT4All

model_name = 'Meta-Llama-3-8B-Instruct.Q4_0.gguf'
#model_name = 'orca-mini-3b-gguf2-q4_0.gguf'

model = GPT4All(model_name=model_name) # allow_download=False for offline mode

while True:
    prompt = input("\n>> User: ")
    with model.chat_session():
        response = model.generate(prompt=prompt, max_tokens=4096, temp=0.7, top_k=40, top_p=0.4, min_p=0.0, repeat_penalty=1.18)
    print(response)