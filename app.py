import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 1. Page Configuration
st.set_page_config(page_title="TechSum: AI & Cyber Summarizer", page_icon="🤖")
st.title("🛡️ TechSum")
st.subheader("Abstractive Summarization for AI & Cybersecurity News")

# 2. Load the Ultimate Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    model_path = "neeltx/TechSum-Ultimate"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# 3. User Input Options
st.write("Upload a text file or paste an article below to get a concise, context-aware summary.")

uploaded_file = st.file_uploader("Upload an article (.txt)", type=("txt"))
text_input = st.text_area("Or paste article text here:", height=300)

article_content = ""
if uploaded_file is not None:
    article_content = uploaded_file.read().decode("utf-8")
elif text_input:
    article_content = text_input

# 4. Summarization Logic
if st.button("Generate Summary"):
    if article_content:
        with st.spinner("TechSum is thinking..."):
            
            # Tokenize the input text
            inputs = tokenizer(article_content, max_length=1024, truncation=True, return_tensors="pt").to(device)
            
            # Generate the summary tokens
            summary_ids = model.generate(
                inputs['input_ids'], 
                do_sample=True,          # Kills the "safe" copy-paste path
                temperature=0.8,         # Turns up creativity
                top_p=0.9,               # Restricts vocabulary to logical next words
                repetition_penalty=1.5,  # Punishes repetitive phrasing
                max_length=150, 
                min_length=50            # Forces a substantive summary
            )
            
            # Decode the tokens back into English
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # --- The Post-Processing Filter (BBC Voice Killer) ---
            lower_summary = summary.lower()
            bad_phrases = ["read more", "find out more", "in this series", "for more on this"]

            for phrase in bad_phrases:
                if phrase in lower_summary:
                    # Find where the bad phrase starts
                    phrase_index = lower_summary.find(phrase)
                    # Keep only the text before the bad phrase
                    summary = summary[:phrase_index].strip()
            # -----------------------------------------------------
            
            # Display Results
            st.success("Summary Generated!")
            st.markdown("### Summary")
            st.write(summary)
            
            st.divider()
            st.info(f"Model used: BART-base (Fine-tuned) | Device: {device}")
    else:
        st.warning("Please provide an article first!")