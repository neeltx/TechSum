import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer

# 1. Load the raw dataset
input_file = 'training_dataset.csv'
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")

# 2. Data Cleaning
# Drop missing values using the correct column names
df = df.dropna(subset=['full_text', 'ground_truth_summary'])
print(f"Rows remaining after cleaning: {len(df)}")

# 3. Exploratory Data Analysis (EDA) - Word Counts
# Calculate lengths to check for outliers
df['article_word_count'] = df['full_text'].apply(lambda x: len(str(x).split()))
df['summary_word_count'] = df['ground_truth_summary'].apply(lambda x: len(str(x).split()))

print("\n--- Basic Statistics ---")
print(df[['article_word_count', 'summary_word_count']].describe())

# 4. Visualization for your Data Analysis Report
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['article_word_count'], bins=50, color='blue', kde=True)
plt.title('Distribution of Article Word Counts')
plt.xlabel('Word Count (full_text)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['summary_word_count'], bins=50, color='green', kde=True)
plt.title('Distribution of Summary Word Counts')
plt.xlabel('Word Count (ground_truth_summary)')
plt.ylabel('Frequency')

plt.tight_layout()
# Saving the plot so you can easily drop it into your Phase 3 report
plt.savefig('eda_word_counts.png') 
plt.show()

# 5. Train-Validation-Test Split
# Split 80% for training, 20% for temp (which becomes 10% val / 10% test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 6. Save the preprocessed datasets
train_file = 'train_data.csv'
val_file = 'val_data.csv'
test_file = 'test_data.csv'

# Saving only the necessary columns to keep things clean
cols_to_save = ['title', 'url', 'full_text', 'ground_truth_summary']
train_df[cols_to_save].to_csv(train_file, index=False)
val_df[cols_to_save].to_csv(val_file, index=False)
test_df[cols_to_save].to_csv(test_file, index=False)

print("\n--- Split Complete ---")
print(f"Saved Training Set to:   {train_file} ({len(train_df)} rows)")
print(f"Saved Validation Set to: {val_file} ({len(val_df)} rows)")
print(f"Saved Test Set to:       {test_file} ({len(test_df)} rows)")

# 7. Tokenization Setup (Feature Extraction)
print("\n--- Initializing BART Tokenizer ---")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Quick test to ensure the tokenizer works on your 'full_text' column
sample_article = train_df['full_text'].iloc[0]
tokens = tokenizer(
    sample_article, 
    max_length=1024,      
    truncation=True,      
    padding='max_length'  
)

print("Tokenization successful! Feature extraction is ready for model training.")