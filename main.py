import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
data_path = "data/IMDB Dataset.csv"
df = pd.read_csv(data_path)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Data Preprocessing
df['cleaned_review'] = df['review'].str.lower()
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
)

# Create DataLoader
train_dataset = ReviewDataset(X_train.tolist(), y_train.tolist(), tokenizer)
test_dataset = ReviewDataset(X_test.tolist(), y_test.tolist(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Load BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train Model
def train(model, train_loader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

train(model, train_loader, optimizer)

# Evaluate Model
def evaluate(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print(classification_report(true_labels, predictions))

evaluate(model, test_loader)

# Save Model
os.makedirs("models", exist_ok=True)
model_save_path = "models/bert_model.pth"
torch.save(model.state_dict(), model_save_path)
print("✅ Model saved successfully at", model_save_path)

# # Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from src.data_preprocessing import clean_text
# from src.feature_extraction import extract_features
# from src.model_training import train_and_save_model, evaluate_model
# from src.utils import load_model
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# import os

# # Load Dataset
# data_path = "data/IMDB Dataset.csv"
# df = pd.read_csv(data_path)

# # Check Data
# print("Sample Data:")
# print(df.head())

# # Data Preprocessing
# df['cleaned_review'] = df['review'].apply(clean_text)

# # Split Data
# X = df['cleaned_review']
# y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature Extraction
# X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)

# # Model Paths
# models_dir = "models"
# os.makedirs(models_dir, exist_ok=True)

# # Logistic Regression Model
# logistic_model = LogisticRegression(max_iter=1000)
# logistic_model_path = os.path.join(models_dir, "logistic_regression.pkl")
# train_and_save_model(logistic_model, X_train_tfidf, y_train, logistic_model_path)
# logistic_model_loaded = load_model(logistic_model_path)
# print("\nLogistic Regression Results:")
# evaluate_model(logistic_model_loaded, X_test_tfidf, y_test)

# # Naive Bayes Model
# nb_model = MultinomialNB()
# nb_model_path = os.path.join(models_dir, "naive_bayes.pkl")
# train_and_save_model(nb_model, X_train_tfidf, y_train, nb_model_path)
# nb_model_loaded = load_model(nb_model_path)
# print("\nNaive Bayes Results:")
# evaluate_model(nb_model_loaded, X_test_tfidf, y_test)

# # Support Vector Machine (SVM) Model
# svm_model = SVC(kernel='linear')
# svm_model_path = os.path.join(models_dir, "svm_model.pkl")
# train_and_save_model(svm_model, X_train_tfidf, y_train, svm_model_path)
# svm_model_loaded = load_model(svm_model_path)
# print("\nSupport Vector Machine Results:")
# evaluate_model(svm_model_loaded, X_test_tfidf, y_test)

# # Sample Prediction Function
# def predict_review(review, model_path, vectorizer):
#     cleaned_review = clean_text(review)
#     vectorized_review = vectorizer.transform([cleaned_review])
#     model = load_model(model_path)
#     prediction = model.predict(vectorized_review)[0]
#     return "Positive" if prediction == 1 else "Negative"

# # Test on a Sample Review
# sample_review = "The movie was absolutely fantastic and I loved every part of it!"
# result = predict_review(sample_review, logistic_model_path, vectorizer)
# print(f"\nSample Review Prediction (Logistic Regression): {result}")
