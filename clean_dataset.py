import pandas as pd
import string

# 1. Load the dataset
df = pd.read_csv('training_dataset.csv')
initial_len = len(df)
df = df.dropna(subset=['full_text', 'ground_truth_summary'])
print(f"Loaded {initial_len} rows. (Dropped {initial_len - len(df)} empty rows)")

# 2. Define the Lead Bias Checker
def check_lead_bias(row):
    text = str(row['full_text']).lower()
    summary = str(row['ground_truth_summary']).lower()
    
    # Remove basic punctuation to ensure clean string matching
    translator = str.maketrans('', '', string.punctuation)
    text_clean = text.translate(translator)
    summary_clean = summary.translate(translator)
    
    text_words = text_clean.split()
    summary_words = summary_clean.split()
    
    # Rule 1: Is the summary suspiciously short? (Less than 10 words)
    if len(summary_words) < 10:
        return True 
        
    # Rule 2: Do the first 10 words of the summary perfectly match the first 10 words of the article?
    check_length = min(len(summary_words), 10)
    text_lead = " ".join(text_words[:check_length])
    summary_lead = " ".join(summary_words[:check_length])
    
    if summary_lead in text_lead or text_lead in summary_lead:
        return True
        
    return False

# 3. Apply the Audit
print("\nAuditing for Lead Bias (Summaries that just copy the first sentence)...")
df['is_bad_data'] = df.apply(check_lead_bias, axis=1)

# 4. Separate the Good from the Bad
bad_data_df = df[df['is_bad_data'] == True]
clean_df = df[df['is_bad_data'] == False]

print(f"\n--- Audit Results ---")
print(f"Detected and removed {len(bad_data_df)} articles with Lead Bias.")
print(f"Remaining high-quality, abstractive examples: {len(clean_df)}")

# 5. Save the fixed datasets
clean_df.drop(columns=['is_bad_data']).to_csv('training_dataset_cleaned.csv', index=False)
bad_data_df.drop(columns=['is_bad_data']).to_csv('discarded_data.csv', index=False)

print("\nSaved the good data to 'training_dataset_cleaned.csv'.")
print("Saved the removed data to 'discarded_data.csv' for your project report.")