from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

train_dataset = Dataset.load_from_disk("rag_dataset/train")
test_dataset = Dataset.load_from_disk("rag_dataset/test")


model = AutoModelForCausalLM.from_pretrained("/from_s3/model", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/from_s3/model")


training_args = TrainingArguments(output_dir = "./output", 
                                  per_device_train_batch_size=4,
                                  evaluation_strategy='steps',
                                  eval_steps=50,
                                  num_train_epochs=1,
                                  report_to="wandb",
                                  logging_steps=1,
                                  learning_rate=1e-5
                                 )

peft_config = LoraConfig(
    r=8,
    task_type="CAUSAL_LM",
    target_modules=["o_proj", "qkv_proj"]
)

def formatting_prompts_func(example):
    output_texts = []
    formatted_dialog = ""
    for question, retrieved, answer in zip(example['question'], example['retrieved'], example['answer']):
        formatted_dialog += f"### User: {question}<end>\t"
        # for RAG
        formatted_dialog += f"### Retrieved: {retrieved}<end>\t"
        formatted_dialog += f"### Bot: {answer}<end>\t"
        output_texts.append(formatted_dialog)

    return output_texts


trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_seq_length=512,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

trainer.train()
trainer.save_model("/app/output")