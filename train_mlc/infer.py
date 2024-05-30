from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from peft import PeftModel
from datasets import Dataset
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("/from_s3/model", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/from_s3/model")
peft_model = PeftModel.from_pretrained(model, '/from_s3/adapter').to('cuda')
test_data = Dataset.load_from_disk('/app/rag_dataset/test')


template = "### User: {question}<end>\t### Bot: "
outputs = []

eos_token_id = [tokenizer.encode("<", add_special_tokens=False)[0], 
                tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0],
                tokenizer.encode("<|end|>", add_special_tokens=False)[0],
                ]

for i, sample in tqdm(enumerate(test_data)):
    input = tokenizer(template.format(question=sample['question']), return_tensors='pt').to(peft_model.device)
    output = peft_model.generate(**input, max_new_tokens=512, repetition_penalty=1.2, eos_token_id=eos_token_id)
    outputs.append(tokenizer.batch_decode(output)[0])
    print(f"n={i}\nquestion={sample['question']}\nanswer={outputs[-1]}")

with open('/app/output/outputs.txt', 'w') as f:
    for line in outputs:
        f.write(line + '\n')