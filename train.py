import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from torch.utils.data import Dataset

# 1. Hardware Check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Custom Dataset Class for PyTorch
class TechSumDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024, target_max_length=150):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        article = str(row['full_text'])
        summary = str(row['ground_truth_summary'])

        # Tokenize the input article
        inputs = self.tokenizer(
            article, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        # Tokenize the target summary
        targets = self.tokenizer(
            summary, max_length=self.target_max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# 3. Load Tokenizer and Base Model
print("\nLoading BART model...")
tokenizer = BartTokenizer.from_pretrained('./local-bart-base')
model = BartForConditionalGeneration.from_pretrained('./local-bart-base').to(device)

# 4. Load Preprocessed Data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')

train_dataset = TechSumDataset(train_df, tokenizer)
val_dataset = TechSumDataset(val_df, tokenizer)

# 5. Define Training Strategy (Hyperparameters)
training_args = TrainingArguments(
    output_dir='./techsum_checkpoints',
    num_train_epochs=3,              # Train by passing over the data 3 times
    per_device_train_batch_size=4,   # Process 4 articles at a time
    per_device_eval_batch_size=4,
    learning_rate=2e-5,              # How aggressively the model updates weights
    weight_decay=0.01,               # Helps prevent overfitting
    logging_dir='./logs',            # Saves the required Training Logs
    logging_steps=10,                
    eval_strategy="epoch",           # Evaluate validation loss every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"                 # Keeps console output clean
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 7. Execute Training
print("\nStarting Model Training... (This may take a while)")
trainer.train()

# 8. Save Final Fine-Tuned Model
final_model_path = "./techsum_final_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nTraining complete! Your custom model is saved to {final_model_path}")