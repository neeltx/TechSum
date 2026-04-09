import newspaper
from newspaper import Config
import pandas as pd
import time
import os

# 1. DEFINE HIGH-VOLUME DATA SOURCES
# These sub-category URLs are chosen for their high article density and 
# high-quality editorial summaries in the meta-tags.
NEWS_SOURCES = [
    "https://www.bleepingcomputer.com/news/security/",
    "https://www.bleepingcomputer.com/news/microsoft/",
    "https://www.securityweek.com/cybercrime-hacking",
    "https://www.securityweek.com/malware-vulnerabilities",
    "https://www.darkreading.com/vulnerabilities-threats",
    "https://www.darkreading.com/endpoint",
    "https://www.infosecurity-magazine.com/news/",
    "https://www.helpnetsecurity.com/category/news/",
    "https://techcrunch.com/category/security/",
    "https://www.zdnet.com/topic/security/",
    "https://www.theregister.com/security/",
    "https://www.csoonline.com/news/"
]

# 2. CONFIGURE SCRAPER SETTINGS
config = Config()
# Using a standard browser User-Agent to avoid being flagged as a bot
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
# Force the scraper to re-check pages for new links
config.memoize_articles = False 
config.fetch_images = False
config.request_timeout = 10

OUTPUT_FILE = "technical_news_dataset.csv"

def collect_data():
    print(f"[*] Starting collection. Target: {OUTPUT_FILE}")
    
    # Track progress in the current session
    new_articles_count = 0
    
    for source_url in NEWS_SOURCES:
        print(f"\n[*] Scanning Category: {source_url}")
        try:
            # Build the list of article links from the source
            paper = newspaper.build(source_url, config=config)
            print(f"    - Found {len(paper.articles)} potential articles.")
            
            # Process up to 100 articles per category to ensure diversity
            for article in paper.articles[:100]:
                try:
                    article.download()
                    article.parse()
                    
                    title = article.title
                    url = article.url
                    full_text = article.text
                    # Extract the meta-description to serve as the ground truth summary
                    summary = article.meta_description
                    
                    # DATA QUALITY FILTERS
                    # Only keep articles with a substantial body and a valid summary
                    if not summary or len(summary.split()) < 10:
                        continue
                    if not full_text or len(full_text.split()) < 100:
                        continue
                        
                    # Prepare the data for CSV
                    data = {
                        'title': [title],
                        'url': [url],
                        'ground_truth_summary': [summary],
                        'full_text': [full_text]
                    }
                    
                    df_new = pd.DataFrame(data)
                    
                    # Append directly to the CSV file
                    # If the file doesn't exist, write headers; otherwise, skip them
                    file_exists = os.path.isfile(OUTPUT_FILE)
                    df_new.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists, encoding='utf-8')
                    
                    new_articles_count += 1
                    if new_articles_count % 10 == 0:
                        print(f"    [+] Progress: {new_articles_count} articles saved...")
                    
                    # Polite delay to prevent server overload
                    time.sleep(0.5)
                    
                except Exception as e:
                    # Skip individual articles that fail to download or parse
                    continue
                    
        except Exception as e:
            print(f"    [!] Error building source {source_url}: {e}")
            continue

    print(f"\n[!] Session Complete. Total new articles added: {new_articles_count}")

def verify_dataset():
    """Checks the total unique entries in the dataset."""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        # Drop duplicates based on full text to see the true unique article count
        unique_df = df.drop_duplicates(subset=['full_text'])
        print(f"--- DATASET STATUS ---")
        print(f"Total Rows in CSV: {len(df)}")
        print(f"Unique Articles:   {len(unique_df)}")
        print(f"----------------------")
    else:
        print("[!] No dataset file found yet.")

if __name__ == "__main__":
    collect_data()
    verify_dataset()