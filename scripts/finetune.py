from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def finetune_model(model_name="microsoft/deberta-v3-base", output_dir="../models/finetuned_sentiment_model"):
    # Load the tokenizer and base model in 4-bit mode
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    # QLoRA configuration (adjust target_modules and hyperparameters as needed)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["classifier"],  # Adjust if needed
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    # Load sentiment dataset (e.g., SST-2)
    dataset = load_dataset("sst2")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Example: Load base AutoModel (not for classification)
    base_model = AutoModel.from_pretrained(model_name)
    print("Base model loaded for other tasks (e.g., embeddings, fill-mask, etc.)")


if __name__ == "__main__":
    finetune_model()
