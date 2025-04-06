# SHL Grammar Scoring Engine - Advanced Solution

# ----------------------------------------
# 🛠️ Libraries Installation
# ----------------------------------------
!pip install -q openai-whisper transformers sentence-transformers language-tool-python spacy pandas scikit-learn xgboost librosa matplotlib seaborn nltk textstat
!python -m spacy download en_core_web_sm
!python -m nltk.downloader punkt vader_lexicon

# ----------------------------------------
# 📂 Data Loading and Exploration
# ----------------------------------------
import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")
sample_sub = pd.read_csv("/content/sample_submission.csv")

# Explore data
print(f"Training data: {train_df.shape[0]} samples")
print(f"Test data: {test_df.shape[0]} samples")
print("\nTraining data preview:")
print(train_df.head())
print("\nLabel distribution:")
print(train_df['label'].describe())

# Visualize distribution of scores
plt.figure(figsize=(10, 6))
sns.histplot(train_df['label'], bins=10, kde=True)
plt.title('Distribution of Grammar Scores in Training Data')
plt.xlabel('Grammar Score')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------
# 🎧 Audio Transcription with Whisper
# ----------------------------------------
import whisper
import time

def transcribe_audio(path, model):
    """Transcribe audio file using Whisper ASR"""
    try:
        result = model.transcribe(path)
        return result['text']
    except Exception as e:
        print(f"Error transcribing {path}: {e}")
        return ""

# Load Whisper model - using base for speed/accuracy tradeoff
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!")

# ----------------------------------------
# 📊 Enhanced Feature Extraction
# ----------------------------------------
import language_tool_python
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from nltk.tokenize import word_tokenize

# Initialize NLP tools
print("Loading NLP tools...")
nlp = spacy.load("en_core_web_sm")
language_tool = language_tool_python.LanguageTool('en-US')
sia = SentimentIntensityAnalyzer()
print("NLP tools loaded successfully!")

def extract_linguistic_features(text):
    """Extract comprehensive linguistic features from text"""
    if not text or text.strip() == "":
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Basic counts
    doc = nlp(text)
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    
    # Grammar errors
    errors = language_tool.check(text)
    num_errors = len(errors)
    error_density = num_errors / len(text) if len(text) > 0 else 0
    
    # Token statistics
    num_tokens = len(doc)
    avg_token_len = sum(len(token.text) for token in doc) / num_tokens if num_tokens > 0 else 0
    
    # Part of speech counts
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    
    noun_ratio = pos_counts.get('NOUN', 0) / num_tokens if num_tokens > 0 else 0
    verb_ratio = pos_counts.get('VERB', 0) / num_tokens if num_tokens > 0 else 0
    adj_ratio = pos_counts.get('ADJ', 0) / num_tokens if num_tokens > 0 else 0
    
    # Readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    
    # Text complexity
    avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
    
    # Named entities
    num_entities = len(doc.ents)
    
    # Sentiment
    sentiment = sia.polarity_scores(text)
    sentiment_score = sentiment['compound']
    
    # Uniqueness of vocabulary
    unique_words = set(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct)
    lexical_diversity = len(unique_words) / num_tokens if num_tokens > 0 else 0
    
    return pd.Series([
        num_sentences, 
        num_errors, 
        error_density,
        num_tokens, 
        avg_token_len,
        noun_ratio,
        verb_ratio,
        adj_ratio,
        flesch_reading_ease,
        flesch_kincaid_grade,
        avg_sentence_length,
        num_entities,
        sentiment_score,
        lexical_diversity
    ])

# ----------------------------------------
# 🔡 Advanced Text Embeddings
# ----------------------------------------
from sentence_transformers import SentenceTransformer

print("Loading sentence transformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence transformer model loaded successfully!")

def get_text_embedding(text):
    """Generate embedding for text using Sentence Transformer"""
    if not text or text.strip() == "":
        # Return zero vector with correct dimensions if text is empty
        return np.zeros(embedder.get_sentence_embedding_dimension())
    return embedder.encode(text)

# ----------------------------------------
# 🧪 Complete Feature Processing Pipeline
# ----------------------------------------
def process_dataset(df, audio_dir, is_training=True):
    """Process entire dataset: transcribe audio, extract features, and create embeddings"""
    # Step 1: Transcribe audio files
    print(f"Transcribing {len(df)} audio files...")
    texts = []
    for f in tqdm(df['filename']):
        audio_path = os.path.join(audio_dir, f)
        text = transcribe_audio(audio_path, whisper_model)
        texts.append(text)
    
    df['transcript'] = texts
    print("Transcription complete!")
    
    # Step 2: Extract linguistic features
    print("Extracting linguistic features...")
    feature_names = [
        'num_sentences', 'num_errors', 'error_density', 'num_tokens', 'avg_token_len',
        'noun_ratio', 'verb_ratio', 'adj_ratio', 'flesch_reading_ease', 'flesch_kincaid_grade',
        'avg_sentence_length', 'num_entities', 'sentiment_score', 'lexical_diversity'
    ]
    
    features = pd.DataFrame([extract_linguistic_features(text) for text in tqdm(df['transcript'])])
    features.columns = feature_names
    
    # Step 3: Generate text embeddings
    print("Generating text embeddings...")
    embeddings = np.vstack([get_text_embedding(text) for text in tqdm(df['transcript'])])
    
    # Step 4: Combine all features
    embedding_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    all_features = pd.concat([features.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)
    
    # Print feature stats if in training mode
    if is_training:
        print("\nFeature statistics:")
        print(all_features[feature_names].describe())
        
        # Feature correlations with target
        if 'label' in df.columns:
            corrs = pd.DataFrame({
                'feature': feature_names,
                'correlation': [all_features[f].corr(df['label']) for f in feature_names]
            }).sort_values('correlation', ascending=False)
            
            print("\nTop feature correlations with target:")
            print(corrs)
            
            # Visualize top correlations
            plt.figure(figsize=(10, 6))
            sns.barplot(x='correlation', y='feature', data=corrs)
            plt.title('Feature Correlations with Grammar Score')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    return all_features

# ----------------------------------------
# 🧠 Advanced Model Training
# ----------------------------------------
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def train_evaluate_models(X, y):
    """Train multiple models and evaluate them"""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to try
    models = {
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_preds = model.predict(X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, val_preds)
        pearson = pearsonr(y_val, val_preds)[0]
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'pearson': pearson
        }
        
        print(f"{name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Pearson: {pearson:.4f}")
        
        # Visualize predictions
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_val, y=val_preds, alpha=0.6)
        sns.regplot(x=y_val, y=val_preds, scatter=False, color='red')
        
        # Plot perfect prediction line
        min_val = min(y_val.min(), val_preds.min())
        max_val = max(y_val.max(), val_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
        
        plt.xlabel("Actual Grammar Score")
        plt.ylabel("Predicted Grammar Score")
        plt.title(f"{name} Model Predictions (Pearson: {pearson:.4f})")
        plt.grid(alpha=0.3)
        plt.show()
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['pearson'])
    best_model = results[best_model_name]['model']
    best_score = results[best_model_name]['pearson']
    
    print(f"\nBest model: {best_model_name} (Pearson: {best_score:.4f})")
    
    return best_model, results

# ----------------------------------------
# 💡 Feature Importance Analysis
# ----------------------------------------
def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importances from the model"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Get top 20 features
        top_indices = indices[:20]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), top_importances, align='center')
        plt.yticks(range(len(top_indices)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.show()

# ----------------------------------------
# 🚀 Main Execution Pipeline
# ----------------------------------------
def main():
    """Main execution pipeline"""
    print("=" * 50)
    print("Starting SHL Grammar Scoring Engine")
    print("=" * 50)
    
    # Process training data
    print("\nProcessing training data...")
    train_features = process_dataset(train_df, "/content/audio_train", is_training=True)
    train_labels = train_df['label']
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    best_model, model_results = train_evaluate_models(train_features, train_labels)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_names = train_features.columns.tolist()
    analyze_feature_importance(best_model, feature_names)
    
    # Process test data
    print("\nProcessing test data...")
    test_features = process_dataset(test_df, "/content/audio_test", is_training=False)
    
    # Generate predictions
    print("\nGenerating predictions for test data...")
    test_preds = best_model.predict(test_features)
    
    # Adjust predictions to match expected range (0-5)
    test_preds = np.clip(test_preds, 0, 5)
    
    # Create submission file
    print("\nCreating submission file...")
    sample_sub['label'] = test_preds
    submission_path = "submission.csv"
    sample_sub.to_csv(submission_path, index=False)
    
    print(f"\nSubmission file saved to {submission_path}")
    print("\nSummary of predictions:")
    print(pd.DataFrame(test_preds).describe())
    
    print("\nSHL Grammar Scoring Engine completed successfully!")

# Run the main pipeline
if __name__ == "__main__":
    main()
