import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from torch.utils.data import Dataset
import optuna

# 1. Hardware Check
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Custom Dataset Class (Same as before)
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

# 3. Initialization Functions
tokenizer = BartTokenizer.from_pretrained('./local-bart-base')

# CRITICAL: For automated tuning, we must use a model_init function 
# so Optuna gets a fresh, untrained model for every new test run.
def model_init():
    return BartForConditionalGeneration.from_pretrained('./local-bart-base')

# Load Data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
train_dataset = TechSumDataset(train_df, tokenizer)
val_dataset = TechSumDataset(val_df, tokenizer)

# 4. Define the Search Space
# This function tells Optuna which parameters to scramble and test
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
    }

# 5. Base Training Arguments
# We optimize for 'eval_loss'. Lower loss correlates strongly with higher ROUGE scores,
# but calculates 10x faster than generating full text summaries during every trial.
training_args = TrainingArguments(
    output_dir='./techsum_tune_checkpoints',
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs_tune',
    report_to="none" 
)

# 6. Initialize Trainer with model_init
trainer = Trainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init,
)

# 7. Run the Automated Search
print("\n--- Starting Optuna Hyperparameter Search ---")
print("This will train several models back-to-back. Grab a coffee!")

best_run = trainer.hyperparameter_search(
    direction="minimize",    # We want to minimize validation loss
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=5               # Tests 5 different combinations. Increase if you have time.
)

# 8. Output the Best Results
print("\n=== OPTIMIZATION COMPLETE ===")
print("The best hyperparameters for your dataset are:")
for key, value in best_run.hyperparameters.items():
    print(f"  - {key}: {value}")

print("\nTake these exact values and update your original train.py script to train your ultimate model!")