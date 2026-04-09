import pandas as pd
import string

df = pd.read_csv('training_dataset_cleaned.csv')
initial_len = len(df)

def clean_string(s):
    translator = str.maketrans('', '', string.punctuation)
    return str(s).lower().translate(translator)

def check_heavy_extraction(row, n_gram_size=8):
    text = clean_string(row['full_text']).split()
    summary = clean_string(row['ground_truth_summary']).split()
    
    if len(summary) < n_gram_size:
        return True # Too short to be a good abstractive summary
        
    text_joined = " ".join(text)
    
    # Create a sliding window of n-grams from the summary
    for i in range(len(summary) - n_gram_size + 1):
        ngram = " ".join(summary[i:i + n_gram_size])
        
        # If this exact 8-word chunk exists anywhere in the main text, flag it
        if ngram in text_joined:
            return True
            
    return False

print("Running deep n-gram extraction audit...")
df['is_extractive'] = df.apply(check_heavy_extraction, axis=1)

extractive_df = df[df['is_extractive'] == True]
abstractive_df = df[df['is_extractive'] == False]

print(f"\n--- Deep Audit Results ---")
print(f"Original rows tested: {initial_len}")
print(f"Failed (Heavy copy-pasting detected anywhere): {len(extractive_df)}")
print(f"Passed (Truly abstractive/paraphrased): {len(abstractive_df)}")

# Only save if we actually have enough data left to train on
if len(abstractive_df) > 100:
    abstractive_df.drop(columns=['is_extractive']).to_csv('strict_abstractive_data.csv', index=False)
    print("\nSaved truly abstractive data to 'strict_abstractive_data.csv'.")
else:
    print("\nWARNING: You have too few abstractive examples left to train a deep learning model.")