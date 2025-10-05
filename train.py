from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch

def preprocess_function(examples, tokenizer):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model():
    # Load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # Preprocess data
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_steps=10_000,
        save_strategy="steps",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained("./bart-xsum")
    tokenizer.save_pretrained("./bart-xsum")

if __name__ == "__main__":
    train_model()
