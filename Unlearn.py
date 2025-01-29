import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import pipeline
# from peft import LoraConfig, get_peft_model


def load_model_and_tokenizer(model_path):
    """
    Loads the tokenizer and model from the specified path.

    Args:
        model_path (str): Path to the model directory.

    Returns:
        tuple: (tokenizer, model) objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
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
    # Load dataset from CSV
    import pandas as pd
    df = pd.read_csv(dataset_path)
    hf_dataset = Dataset.from_pandas(df)

    # Tokenize the dataset
    def tokenize_function(example):
        input_encodings = tokenizer(
            example['input'], truncation=True, padding="max_length", max_length=max_length
        )
        output_encodings = tokenizer(
            example['output'], truncation=True, padding="max_length", max_length=max_length
        )

        # Replace pad token IDs with -100 in the labels
        labels = [-100 if token == tokenizer.pad_token_id else token for token in output_encodings['input_ids']]

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
        }

    # Apply tokenization to the dataset
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=['input', 'output'])
    return tokenized_dataset



def finetune(dataset_path, enableLoRA = False):
    '''
    dataset -> 2 columns -> input and output columns

    this should load a local model from Model/Untrained/ directory
    and fine-tune it on the dataset
    add lora Qlora support later 

    return Nothing -> Store the finetuned model into Model/Finetuned directory
    '''

    # Step 1: Load the untrained model
    tokenizer, model = load_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")

    # Step 2: Tokenize the dataset using the separate function
    tokenized_dataset = tokenize_dataset(dataset_path, tokenizer)

    # Step 3: Define training arguments
    training_args = TrainingArguments(
        output_dir="./Model/Finetuned",
        per_device_train_batch_size=4,  
        num_train_epochs=3,  
        logging_dir="./logs", 
        save_steps=500,  # Save checkpoints every 500 steps
        evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
        learning_rate=5e-5, 
        save_total_limit=2,  # Keep only the last 2 checkpoints
    )

    # Step 4: Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Pass the tokenized dataset
    )

    # Step 5: Fine-tune the model
    trainer.train()

    # Step 6: Save the fine-tuned model
    model.save_pretrained("Model/Finetuned")
    tokenizer.save_pretrained("Model/Finetuned")

    print("Fine-tuning complete. Model saved to Model/Finetuned.")

   


dataset_file = "Data/processed_dataset.csv" 
finetune(dataset_file)

