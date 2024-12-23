# https://github.com/microsoft/DialoGPT
# example script on how to use LLMs directly from Hugging Face
# it also uses chat history as a simple memory for the bot.
# How to run it:
# pipenv shell
# python dialoGPT.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chatbot():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    step = 0
    chat_history = ""
    
    while True:
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        user_input = input("\n>> User:")
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # pretty print last ouput tokens from bot
        print("\n>> DialoGPT: {}".format(bot_response))

        chat_history = chat_history + user_input + bot_response
        #print("\n\nhistory: {}".format(chat_history))
        step += 1


if __name__ == '__main__':
    chatbot()
