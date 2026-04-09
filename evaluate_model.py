import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
from tqdm import tqdm

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Evaluation on: {device}")

# 2. Load Your New Ultimate Model
model_path = "./techsum_ultimate_model" # Ensure this matches your extracted folder name
print(f"Loading model from {model_path}...")
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).to(device)

# 3. Load Your Custom Test Data
print("Loading custom test_data.csv...")
test_df = pd.read_csv('test_data.csv')

# Drop any broken rows just in case
test_df = test_df.dropna(subset=['full_text', 'ground_truth_summary'])

articles = test_df['full_text'].tolist()
ground_truths = test_df['ground_truth_summary'].tolist()
predictions = []

# 4. Generate Summaries
print(f"Generating summaries for {len(articles)} test articles...")
model.eval() # Put model in testing mode

# We use standard beam search here because it's the academic standard for ROUGE benchmarking
for article in tqdm(articles, desc="Summarizing"):
    inputs = tokenizer(str(article), max_length=1024, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # --- The Post-Processing Filter (BBC Voice Killer) ---
# We use lower() to catch variations like "read more" or "Read more"
lower_output = output_text.lower()
bad_phrases = ["read more", "find out more", "in this series", "for more on this"]

for phrase in bad_phrases:
    if phrase in lower_output:
        # Find the exact index where the bad phrase starts in the original text
        phrase_index = lower_output.find(phrase)
        # Slice the original text to only keep everything before the bad phrase
        output_text = output_text[:phrase_index].strip()
# -----------------------------------------------------

# Now display output_text in your Streamlit app as usual!
    predictions.append(output_text)

# 5. Calculate ROUGE Scores
print("\nCalculating ROUGE Scores...")
rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predictions, references=ground_truths)

print("\n" + "="*50)
print("🏆 FINAL DOMAIN TRANSFER ROUGE SCORES 🏆")
print("="*50)
print(f"ROUGE-1 (Word Match):      {results['rouge1'] * 100:.2f}%")
print(f"ROUGE-2 (Phrase Match):    {results['rouge2'] * 100:.2f}%")
print(f"ROUGE-L (Sentence Flow):   {results['rougeL'] * 100:.2f}%")
print("="*50)

# 6. The Visual Proof
print("\n--- VISUAL PROOF (First Article) ---")
print(f"ORIGINAL START: {articles[0][:150]}...")
print(f"\nGROUND TRUTH: {ground_truths[0]}")
print(f"\nMODEL OUTPUT: {predictions[0]}")
print("===================================\n")