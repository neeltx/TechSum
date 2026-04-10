from huggingface_hub import login
from transformers import BartTokenizer, BartForConditionalGeneration

# 1. Authenticate (Bypassing the Windows terminal bug)
# Paste your token inside the quotes below!
hf_token = "YOUR_HF_TOKEN_HERE"
login(token=hf_token)

# 2. Setup paths
local_model_path = "./techsum_ultimate_model"

# REPLACE THIS with your actual Hugging Face username!
hf_username = "neeltx" 
repo_name = f"{hf_username}/TechSum-Ultimate"

print(f"\nLoading local model from {local_model_path}...")
tokenizer = BartTokenizer.from_pretrained(local_model_path)
model = BartForConditionalGeneration.from_pretrained(local_model_path)

print(f"Pushing to Hugging Face Hub: {repo_name}...")
# This will take a few minutes depending on your upload speed
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print("🚀 Success! Your model is live on Hugging Face.")