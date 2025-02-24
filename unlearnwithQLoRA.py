import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model_and_tokenizer(model_path, use_qlora=False):
    """
    Loads the tokenizer and model from the specified path, with optional QLoRA configuration.

    Args:
        model_path (str): Path to the model directory.
        use_qlora (bool): Whether to apply QLoRA.

    Returns:
        tuple: (tokenizer, model) objects.
    """
    access_token = ""
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if use_qlora:
        # Load quantized model with 4-bit precision
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  
            bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
            bnb_4bit_compute_dtype="float16",  # Compute in fp16
            bnb_4bit_use_double_quant=True,  # Double quantization
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            token=access_token
        )

        # Prepare model for efficient fine-tuning
        model = prepare_model_for_kbit_training(model)

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,  # Rank
            lora_alpha=32,  
            lora_dropout=0.1,  
            bias="none",
            task_type="CAUSAL_LM",  
            target_modules=["q_proj", "v_proj"],  # Apply LoRA on key attention layers
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)

    return tokenizer, model


def tokenize_dataset(dataset_path, tokenizer, max_length=128):
    """
    Tokenizes the dataset using the specified tokenizer.

    Args:
        dataset_path (str): Path to the dataset CSV with 'input' and 'output' columns.
        tokenizer: Hugging Face tokenizer object.
        max_length (int): Maximum length for tokenization.

    Returns:
        Dataset: Tokenized Hugging Face Dataset.
    """
    df = pd.read_csv(dataset_path)
    hf_dataset = Dataset.from_pandas(df)

    def tokenize_function(example):
        input_encodings = tokenizer(
            example['input'], truncation=True, padding="max_length", max_length=max_length
        )
        output_encodings = tokenizer(
            example['output'], truncation=True, padding="max_length", max_length=max_length
        )

        labels = [-100 if token == tokenizer.pad_token_id else token for token in output_encodings['input_ids']]

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
        }

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=['input', 'output'])
    return tokenized_dataset


def finetune(dataset_path, enableLoRA=False):
    """
    Fine-tunes a local model from the "mistralai/Mistral-7B-Instruct-v0.3" directory.

    Args:
        dataset_path (str): Path to dataset.
        enableLoRA (bool): Whether to use QLoRA.

    Returns:
        None. Saves the fine-tuned model.
    """
    # Step 1: Load the untrained model with QLoRA if enabled
    tokenizer, model = load_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.3", use_qlora=enableLoRA)

    # Step 2: Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset_path, tokenizer)

    # Step 3: Define training arguments
    training_args = TrainingArguments(
        output_dir="./Model/Finetuned",
        per_device_train_batch_size=2,  # Lower batch size for QLoRA
        gradient_accumulation_steps=4,  # Reduce memory usage
        num_train_epochs=3,
        logging_dir="./logs",
        save_steps=500,
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # Lower LR for QLoRA
        save_total_limit=2,
        fp16=True,  # Use mixed precision training
        gradient_checkpointing=True,  # Enable for memory efficiency
    )

    # Step 4: Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Step 5: Fine-tune the model
    trainer.train()

    # Step 6: Save the fine-tuned model
    model.save_pretrained("Model/Finetuned")
    tokenizer.save_pretrained("Model/Finetuned")

    print("Fine-tuning complete. Model saved to Model/Finetuned.")


# Example Usage
dataset_file = "Fine-tuning-withLoRA/Data/processed_dataset.csv" 
finetune(dataset_file, enableLoRA=True)
