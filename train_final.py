import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training Ultimate TechSum Model on: {device}")

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

        inputs = self.tokenizer(article, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer(summary, max_length=self.target_max_length, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# 1. Load the Offline Model
print("\nLoading local BART model...")
tokenizer = BartTokenizer.from_pretrained('./local-bart-base')
model = BartForConditionalGeneration.from_pretrained('./local-bart-base').to(device)

# 2. Load Data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
train_dataset = TechSumDataset(train_df, tokenizer)
val_dataset = TechSumDataset(val_df, tokenizer)

# 3. Apply the Optuna Hyperparameters!
training_args = TrainingArguments(
    output_dir='./techsum_ultimate_checkpoints',
    num_train_epochs=3,              
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=4.68685505098148e-05,  # Optuna's exact learning rate
    weight_decay=0.09673818154508972,    # Optuna's exact weight decay
    logging_dir='./logs_ultimate',            
    logging_steps=10,                
    eval_strategy="epoch",           
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"                 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nStarting Final Model Training...")
trainer.train()

# 4. Save the Ultimate Model
final_model_path = "./techsum_ultimate_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nTraining complete! Your ultimate model is saved to {final_model_path}")