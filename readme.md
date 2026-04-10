# 🛡️ TechSum: Abstractive AI & Cyber News Summarizer

TechSum is a custom Natural Language Processing (NLP) pipeline and web application that generates highly abstractive, context-aware summaries of complex Cybersecurity and Artificial Intelligence news articles. 

## 🚀 The Architecture
1. **Data Engineering:** Scraped and sanitized domain-specific articles.
2. **The "Lead Bias" Discovery:** Initial fine-tuning on custom data revealed a pre-trained "Extractive Lead Bias" (the model heavily relied on copy-pasting the first sentence of articles).
3. **Zero-Shot Domain Transfer:** Pivoted training to the **XSum** dataset using dual T4 GPUs to force the model to learn true abstraction, then evaluated its zero-shot performance on the custom cyber dataset.
4. **Post-Processing Pipeline:** Implemented a heuristic filter in the Streamlit app to sanitize "Domain Style Leakage" inherited from the training data.

## 🛠️ Tech Stack
* **Modeling:** Hugging Face Transformers, PyTorch, BART-base
* **Training:** Kaggle T4 x2 Accelerators, Mixed Precision (fp16)
* **Metrics:** ROUGE Scores (`evaluate` library)
* **Deployment:** Streamlit (Local UI), Hugging Face Hub (Model Weights Hosting)

---

## 💻 How to Run the App Locally

This project is designed to be easily run on any local machine. The heavy model weights (~500MB) are hosted on the Hugging Face Hub and will automatically download the first time you run the application.

### Prerequisites
* Python 3.8+
* Git

### Step-by-Step Installation

**1. Clone the repository**
`git clone https://github.com/neeltx/TechSum.git`
`cd TechSum`

**2. Create a virtual environment (Recommended)**
*On Windows:*
`python -m venv venv`
`venv\Scripts\activate`

*On Mac/Linux:*
`python3 -m venv venv`
`source venv/bin/activate`

**3. Install the required dependencies**
`pip install -r requirements.txt`

**4. Launch the application**
`streamlit run app.py`

*Note: Upon running the command above, the app will open in your default web browser. If it is your first time running the app, it may take a minute or two to automatically fetch the `neeltx/TechSum-Ultimate` model from the cloud.*

---

## 📂 Project Structure
* `app.py` - The main Streamlit web application.
* `push_model.py` - Utility script used to migrate local model weights to the Hugging Face Hub.
* `evaluate_model.py` - Script for calculating ROUGE evaluation metrics.
* `requirements.txt` - Python dependencies for running the environment.
* `data/` - Cleaned and raw `.csv` files used for evaluation.

## 📊 Evaluation
Tested against a custom ground-truth dataset of AI/Cyber news. The model successfully demonstrates high-level synthesis and rephrasing without relying on simple extractive extraction.