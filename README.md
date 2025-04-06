# SHL-Grammar-Scoring-Engine---Advanced-Solution
🔍 Project Objective
Develop an AI-based Grammar Scoring Engine for spoken English audio samples.


Predict a continuous grammar score between 0 and 5 for each audio file.



🎧 Audio Transcription
Used OpenAI Whisper (base) model to convert audio to text.


Ensures accurate, real-time transcription of 45–60 second clips.



🧠 Linguistic Feature Extraction
Extracted features using:


spaCy: sentence and token analysis, POS tagging.


language-tool-python: grammar error detection.


textstat: readability metrics (Flesch, Kincaid scores).


NLTK Sentiment Analyzer: sentiment score.


Computed metrics like:


Number of sentences, grammar errors, token length


POS ratios (noun, verb, adjective)


Lexical diversity, sentiment score, and more.






🔡 Text Embeddings
Generated semantic embeddings using Sentence-Transformers (all-MiniLM-L6-v2).


Captures deeper linguistic patterns and context in the spoken content.



🏗️ Feature Engineering Pipeline
Combined:


Linguistic features (14+ metrics)


384-dimensional sentence embeddings


Created a complete feature set for modeling.



🤖 Model Training & Evaluation
Trained multiple regression models:


Ridge, Random Forest, Gradient Boosting, XGBoost


Split data into training and validation sets.


Evaluated using:


Pearson correlation


RMSE and MAE metrics


Visualized prediction performance with scatter plots and regression lines.



🔍 Feature Importance
Analyzed top contributing features using feature_importances_ from tree models.


Helped interpret model decision-making.




📦 Final Predictions & Submission
Processed test audio files using the same pipeline.


Generated grammar scores and clipped to the [0, 5] range.


Saved results in submission.csv for Kaggle evaluation.

