###=======================================================================================
### Imports

from datasets import load_dataset
import evaluate
import numpy as np
import os
from peft import get_peft_model, LoraConfig, TaskType
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaModel, RobertaTokenizer, Trainer, TrainingArguments
import torch
from torch import cuda

###=======================================================================================
### Run params

# Static
device = "cuda"
model_name = "roberta-base"

# Run-dependent
lora = "lora" in sys.argv
bitfit = "bitfit" in sys.argv
run_str = ""
if lora:
    run_str += "_lora"
if bitfit:
    run_str += "_bitfit"
out_dir = f"out/roberta{run_str}"

###=======================================================================================
### Functions

# Load data
def load_sst2_dataset():
    dataset = load_dataset("stanfordnlp/sst2")
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.rename_column("sentence", "text")
    return dataset

def initialize_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def tokenize_dataset(dataset):
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

###=======================================================================================
### Main execution

# Load dataset and tokenizer
dataset = load_sst2_dataset()
tokenizer = initialize_tokenizer()

# Split dataset into train, validation, and test
train_test_sets = dataset['train'].train_test_split(test_size = dataset['test'].shape[0], random_seed=42)
train_set = train_test_sets['train']
test_set = train_test_sets['test']
val_set = dataset['validation']

# Tokenize each dataset
tokenized_train_set = tokenize_dataset(train_set)
tokenized_val_set = tokenize_dataset(val_set)
tokenized_test_set = tokenize_dataset(test_set)

## Instantiate Roberta model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="single_label_classification")
model.to(device)

# If doing LoRA, configure model
if lora:
    peft_config = LoraConfig(
        r=4, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        bias="none", 
        task_type=TaskType.SEQ_CLS
        )
    model = get_peft_model(model, peft_config)

# If doing BitFit, freeze all non-bias parameters
if bitfit:
    for name, param in model.named_parameters():
        param.requires_grad = 'bias' in name
    
# Set up training
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
        save_steps=10000,
        save_total_limit=1,
        logging_dir = f"./log_roberta{run_str}",
        output_dir=out_dir, 
        eval_strategy="epoch",
        learning_rate=(1e-04 if bitfit else 1e-05) # https://ming-liu.medium.com/llm-peft-principles-explained-1-bitfit-90f6d6d3c50d
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_set,
    eval_dataset=tokenized_val_set,
    compute_metrics=compute_metrics
)

# Run training
trainer.train()

# Save model
trainer.save_model(out_dir + "/saved_model")

# Evaluate model
val_results = trainer.predict(tokenized_val_set)
test_results = trainer.predict(tokenized_test_set)
print(val_results)
print("validation results:", val_results.metrics)
print(test_results)
print("test results:", test_results.metrics)

