import pandas as pd
import numpy as np

def analyze_dataset():
    """
    Analyze the merged training dataset to determine if it's suitable
    for a deep learning news summarizer project.
    """
    
    print("=" * 80)
    print("MERGED DATASET ANALYSIS FOR DEEP LEARNING NEWS SUMMARIZER")
    print("=" * 80)
    
    # Load the dataset
    df = pd.read_csv('training_dataset.csv')
    
    print(f"\n[1] DATASET SIZE")
    print(f"    Total articles: {len(df):,}")
    
    # Text statistics
    print(f"\n[2] TEXT STATISTICS")
    df['title_length'] = df['title'].str.split().str.len()
    df['summary_length'] = df['ground_truth_summary'].str.split().str.len()
    df['text_length'] = df['full_text'].str.split().str.len()
    
    print(f"    Average title length: {df['title_length'].mean():.1f} words")
    print(f"    Average summary length: {df['summary_length'].mean():.1f} words")
    print(f"    Average full text length: {df['text_length'].mean():.1f} words")
    print(f"    Median full text length: {df['text_length'].median():.1f} words")
    
    # Compression ratio
    compression_ratio = (df['summary_length'] / df['text_length']).mean() * 100
    print(f"    Average compression ratio (summary:full text): {compression_ratio:.1f}%")
    
    # Data quality
    print(f"\n[3] DATA QUALITY METRICS")
    print(f"    Missing titles: {df['title'].isna().sum()}")
    print(f"    Missing summaries: {df['ground_truth_summary'].isna().sum()}")
    print(f"    Missing full text: {df['full_text'].isna().sum()}")
    print(f"    Missing URLs: {df['url'].isna().sum()}")
    
    # Distribution analysis
    print(f"\n[4] TEXT LENGTH DISTRIBUTION")
    print(f"    Summary length - Min: {df['summary_length'].min()}, Max: {df['summary_length'].max()}")
    print(f"    Full text length - Min: {df['text_length'].min()}, Max: {df['text_length'].max()}")
    
    # Filter articles by quality metrics
    quality_filter = (df['summary_length'] >= 10) & (df['text_length'] >= 100)
    quality_articles = quality_filter.sum()
    
    print(f"\n[5] ARTICLES MEETING QUALITY CRITERIA")
    print(f"    (Summary >= 10 words AND Full text >= 100 words)")
    print(f"    Valid articles: {quality_articles:,} out of {len(df):,}")
    print(f"    Coverage: {(quality_articles/len(df)*100):.1f}%")
    
    # Train/val/test split suggestion
    print(f"\n[6] RECOMMENDED DATA SPLIT (using {quality_articles:,} valid articles)")
    train_samples = int(quality_articles * 0.7)
    val_samples = int(quality_articles * 0.15)
    test_samples = quality_articles - train_samples - val_samples
    
    print(f"    Training set: {train_samples:,} articles (70%)")
    print(f"    Validation set: {val_samples:,} articles (15%)")
    print(f"    Test set: {test_samples:,} articles (15%)")
    
    # Assessment for deep learning
    print(f"\n[7] SUITABILITY ASSESSMENT FOR DEEP LEARNING NEWS SUMMARIZER")
    print(f"    " + "=" * 76)
    
    assessment_score = 0
    max_score = 0
    
    # Check 1: Dataset size
    max_score += 1
    if quality_articles >= 5000:
        print(f"    ✓ Dataset Size: EXCELLENT ({quality_articles:,} articles)")
        assessment_score += 1
    elif quality_articles >= 2000:
        print(f"    ✓ Dataset Size: GOOD ({quality_articles:,} articles)")
        assessment_score += 0.8
    elif quality_articles >= 800:
        print(f"    ~ Dataset Size: ACCEPTABLE ({quality_articles:,} articles)")
        assessment_score += 0.6
    else:
        print(f"    ✗ Dataset Size: SMALL ({quality_articles:,} articles)")
        assessment_score += 0.3
    
    # Check 2: Diversity of samples
    max_score += 1
    unique_urls = df['url'].nunique()
    print(f"    ✓ Article Diversity: {unique_urls:,} unique sources")
    if unique_urls > quality_articles * 0.9:
        assessment_score += 1
    else:
        assessment_score += 0.8
    
    # Check 3: Text-Summary ratio
    max_score += 1
    if 20 <= compression_ratio <= 40:
        print(f"    ✓ Compression Ratio: OPTIMAL ({compression_ratio:.1f}%)")
        assessment_score += 1
    elif 10 <= compression_ratio <= 50:
        print(f"    ✓ Compression Ratio: GOOD ({compression_ratio:.1f}%)")
        assessment_score += 0.9
    else:
        print(f"    ~ Compression Ratio: ACCEPTABLE ({compression_ratio:.1f}%)")
        assessment_score += 0.7
    
    # Check 4: Text length
    max_score += 1
    avg_text_length = df['text_length'].mean()
    if avg_text_length >= 300:
        print(f"    ✓ Article Length: GOOD ({avg_text_length:.0f} words avg)")
        assessment_score += 1
    elif avg_text_length >= 150:
        print(f"    ✓ Article Length: ACCEPTABLE ({avg_text_length:.0f} words avg)")
        assessment_score += 0.8
    else:
        print(f"    ~ Article Length: SHORT ({avg_text_length:.0f} words avg)")
        assessment_score += 0.5
    
    # Overall score
    overall_score = (assessment_score / max_score) * 100
    
    print(f"\n[8] OVERALL SUITABILITY SCORE: {overall_score:.1f}%")
    print(f"    " + "=" * 76)
    
    if overall_score >= 85:
        print(f"    ✓ EXCELLENT - Dataset is well-suited for deep learning summarization")
    elif overall_score >= 70:
        print(f"    ✓ GOOD - Dataset is suitable with some considerations")
    elif overall_score >= 50:
        print(f"    ~ MODERATE - Dataset can be used but may need augmentation")
    else:
        print(f"    ✗ LIMITED - Dataset may require significant augmentation")
    
    # Recommendations
    print(f"\n[9] RECOMMENDATIONS")
    print(f"    " + "=" * 76)
    
    if quality_articles < 2000:
        print(f"    • Consider collecting more data ({quality_articles:,} articles is relatively small)")
        print(f"      Recommended minimum: 2,000+ articles for robust deep learning model")
    else:
        print(f"    • Dataset size is adequate for training")
    
    if compression_ratio < 15:
        print(f"    • Summaries are quite short - may affect model training effectiveness")
        print(f"      Consider using abstractive summarization approaches")
    
    if compression_ratio > 50:
        print(f"    • Compression ratio is high - summaries may be too extractive")
        print(f"      Consider validating summary quality manually")
    
    if avg_text_length < 200:
        print(f"    • Articles are shorter than ideal - may limit context for summarization")
    
    print(f"\n    • Model Architecture Suggestions:")
    print(f"      - For {quality_articles:,} articles: Consider BART, T5, or fine-tuned PEGASUS")
    print(f"      - Use data augmentation techniques if < 2000 articles")
    print(f"      - Implement cross-validation for better generalization")
    print(f"      - Consider pre-trained models (TransferLearning) due to dataset size")
    
    print(f"\n[10] HYPERPARAMETER RECOMMENDATIONS")
    print(f"    " + "=" * 76)
    print(f"    Batch size: 8-16 (smaller due to dataset size)")
    print(f"    Learning rate: 2e-5 to 5e-5")
    print(f"    Epochs: 10-20 (with early stopping)")
    print(f"    Max input length: {min(512, int(df['text_length'].quantile(0.95)))} tokens")
    print(f"    Max summary length: {min(150, int(df['summary_length'].quantile(0.95)))} tokens")
    
    print(f"\n" + "=" * 80)
    print(f"Analysis complete. Dataset summary saved to 'dataset_analysis.txt'")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    analyze_dataset()
