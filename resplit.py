import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the STRICT dataset
df = pd.read_csv('strict_abstractive_data.csv')

# 2. Re-split: 80% Train, 10% Val, 10% Test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

cols_to_save = ['title', 'url', 'full_text', 'ground_truth_summary']

# 3. Overwrite the split files
train_df[cols_to_save].to_csv('train_data.csv', index=False)
val_df[cols_to_save].to_csv('val_data.csv', index=False)
test_df[cols_to_save].to_csv('test_data.csv', index=False)

print(f"Deep Clean Re-split complete.")
print(f"Final Training Rows: {len(train_df)}")