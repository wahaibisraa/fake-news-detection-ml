#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:21:55 2024

@author: lamyaalrahbi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords  # Import stopwords
from nltk.tokenize import word_tokenize  # Import tokenizer
import nltk

# Download required resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')


# Load the dataset
data = pd.read_csv("WELFake_Dataset.csv")

# Get an overview of the dataset
print(data.info())
print(data.describe())
# Check for missing values
print(data.isnull().sum())

#  Data Preprocessing
# Combine 'title' and 'text' columns
data['content'] = data['title'].fillna("") + " " + data['text'].fillna("")

# Handle missing values in the target variable
data = data.dropna(subset=['label'])

#Data Cleaning
print(f"Initial dataset size: {len(data)}")
data.dropna(subset=['text', 'label'], inplace=True)
data['content'] = data['title'].fillna("") + " " + data['text'].fillna("")
data = data[data['text'].apply(len) < 10000]
data.drop_duplicates(subset='content', inplace=True)
print(f"Dataset size after cleaning: {len(data)}")

# Target variable
y = data['label']

# Display Correlation Heatmap
# Select only numerical columns for correlation matrix
numerical_data = data.select_dtypes(include=['number'])

# Check if numerical columns exist
if not numerical_data.empty:
   correlation_matrix = numerical_data.corr()
   plt.figure(figsize=(8, 6))
   sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("Correlation Heatmap")
   plt.show()
else:
   print("No numerical columns found for correlation heatmap.")
   
# TF-IDF Vectorization for 'content' column
tfidf = TfidfVectorizer(max_features=1500, stop_words='english')  # Use 1500 features for better representation
X = tfidf.fit_transform(data['content'])

# Step 2: Standardize Features
scaler = StandardScaler(with_mean=False)  # Set `with_mean=False` for sparse matrices
X_scaled = scaler.fit_transform(X)

# Step 3: PCA for Dimensionality Reduction
pca = PCA(n_components=50, random_state=42)  # Reduce to 50 components
X_pca = pca.fit_transform(X_scaled.toarray())  # Convert to dense for PCA

# Visualize PCA (first 2 components)
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("PCA Scatter Plot")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(label='Class')
plt.show()

# Step 4: Model Building with GridSearchCV for Isolation Forest
param_grid = {
   'n_estimators': [100, 200],
   'max_samples': [0.8, 1.0],
   'contamination': [0.1, 0.2],  # Adjust based on the problem
   'random_state': [42]
}

iso_forest = IsolationForest()
grid_search = GridSearchCV(iso_forest, param_grid, scoring='f1_weighted', cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_pca, y)

# Get best model and parameters
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 5: Evaluate Performance
y_pred = best_model.predict(X_pca)

# Convert predictions to binary (1 for normal, 0 for anomaly)
y_pred_binary = [1 if x == 1 else 0 for x in y_pred]

# Classification Report
print("Model Evaluation Metrics:")
print(classification_report(y, y_pred_binary))

# Accuracy, Precision, Recall, F1-score
accuracy = accuracy_score(y, y_pred_binary)
precision = precision_score(y, y_pred_binary, average='weighted')
recall = recall_score(y, y_pred_binary, average='weighted')
f1 = f1_score(y, y_pred_binary, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# User Input for Classification
def classify_text(user_input):
    # Transform the user input text
    input_tfidf = tfidf.transform([user_input])
    input_scaled = scaler.transform(input_tfidf)
    input_pca = pca.transform(input_scaled.toarray())
    
    # Predict using the trained model
    prediction = best_model.predict(input_pca)
    # Convert the prediction to binary (1 for normal, 0 for anomaly)
    prediction_binary = 1 if prediction == 1 else 0
    
    # Return classification result
    if prediction_binary == 1:
        return "This is a real news article."
    else:
        return "This is a fake news article."

# Example usage
user_input = input("Enter the text to classify: ")
result = classify_text(user_input)
print(result)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import nltk


from tqdm import tqdm
from joblib import Parallel, delayed

import seaborn as sns


# Log initial dataset size
print(f"Initial dataset size: {len(data)}")

# Handle missing values
data = data.dropna(subset=['text'])

# Handle outliers in text length

data = data[data['text'].apply(len) < 10000]  # Remove excessively long texts

# Log dataset size after preprocessing
print(f"Dataset size after removing long texts: {len(data)}")

import nltk
nltk.download('punkt')

import nltk
nltk.download('punkt_tab')

nltk.download('punkt')


from joblib import Parallel, delayed
from tqdm import tqdm

def preprocess_text_parallel(data, n_jobs=-1):
    """
    Preprocess text data in parallel using joblib.
    
    Parameters:
    - data (list): List of text data to preprocess.
    - n_jobs (int): Number of parallel jobs to run (-1 uses all available cores).
    
    Returns:
    - list: Preprocessed text data.
    """
    # Ensure stopwords are loaded
    stop_words = set(stopwords.words('english'))

    # Define single text preprocessing
    def preprocess_single_text(text):
        tokens = word_tokenize(text)
        cleaned_tokens = [
            word for word in tokens if word.isalpha() and word not in stop_words
        ]
        return ' '.join(cleaned_tokens)

    # Define chunk size
    chunk_size = 10000
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    results = []

    # Process data in chunks with parallel processing
    for chunk in tqdm(chunks, desc="Preprocessing Text in Chunks"):
        processed_chunk = Parallel(n_jobs=n_jobs, timeout=None)(
            delayed(preprocess_single_text)(text) for text in chunk
        )
        results.extend(processed_chunk)

    return results

# Example Usage
if __name__ == "__main__":
    print("Starting text preprocessing...")
    # Assuming 'data' is a pandas DataFrame with a column 'text'
    data['text_cleaned'] = preprocess_text_parallel(data['text'].tolist(), n_jobs=2)  # Adjust n_jobs as needed
    print("Text preprocessing completed.")

# Save preprocessed data for reuse
data.to_csv("Preprocessed_FakeNews.csv", index=False)

# Define features (X) and target (y)
X = data['text_cleaned']
y = data['label']

# Define custom keywords
custom_keywords = ["breaking", "exclusive", "shocking", "urgent", "alert"]

# Add custom keyword features
def add_keyword_features(df, keywords):
    for keyword in keywords:
        df[f"has_{keyword}"] = df['text_cleaned'].apply(lambda x: 1 if keyword in x.lower() else 0)
    return df

# Add binary features for custom keywords
data = add_keyword_features(data, custom_keywords)

# Split the dataset into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Text preprocessing: TF-IDF vectorization for train and test sets
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)

# Combine TF-IDF with custom keyword features
X_train_keywords = data.loc[X_train_raw.index, [f"has_{kw}" for kw in custom_keywords]].values
X_test_keywords = data.loc[X_test_raw.index, [f"has_{kw}" for kw in custom_keywords]].values
X_train = np.hstack((X_train_tfidf.toarray(), X_train_keywords))
X_test = np.hstack((X_test_tfidf.toarray(), X_test_keywords))

# Perform cross-validation
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
cross_val_scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cross_val_scores}")
print(f"Mean cross-validation accuracy: {cross_val_scores.mean()}")

# Train the model
print("Training Random Forest Classifier...")
rf_clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_rf = rf_clf.predict(X_test)
probs_rf = rf_clf.predict_proba(X_test)[:, 1]  # Probability for ROC curve
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Confusion Matrix 
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], cbar=False)
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.show()

# Feature Importance Bar Plot
feature_importances = rf_clf.feature_importances_
indices = np.argsort(feature_importances)[-20:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), feature_importances[indices], align='center', color='teal')
plt.yticks(range(len(indices)), [vectorizer.get_feature_names_out()[i] for i in indices])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Feature Importances', fontsize=14)
plt.show()

# ROC Curve 
fpr, tpr, _ = roc_curve(y_test, probs_rf)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
plt.legend(loc='lower right')
plt.show()
print(y.value_counts())

import numpy as np
from sklearn.metrics import accuracy_score

# Assuming you have these predictions from both models
# Model 1 (Isolation Forest) predictions
y_pred_model1 = np.array([1, 1, 0, 0, 1])  # Model 1's predictions (1: normal, 0: anomaly)

# Model 2 (Random Forest) predictions
y_pred_model2 = np.array([1, 0, 1, 0, 1])
# Function to combine predictions using majority voting
def final_prediction(pred1, pred2):
    assert len(pred1) == len(pred2), "Predictions must have the same length"
    
    combined_predictions = []
    for p1, p2 in zip(pred1, pred2):
        combined_predictions.append(np.argmax(np.bincount([p1, p2])))
    
    return np.array(combined_predictions)
# Combine predictions
final_predictions = final_prediction(y_pred_model1, y_pred_model2)

# Print the final predictions
print("Final Predictions:", final_predictions)

y_true = np.array([1, 1, 0, 0, 1])  

# Evaluate the final predictions
final_accuracy = accuracy_score(y_true, final_predictions)
print(f"Final Accuracy: {final_accuracy:.4f}")
# print the classification report if needed
from sklearn.metrics import classification_report
print("\nFinal Classification Report:\n", classification_report(y_true, final_predictions))