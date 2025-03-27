from transformers import BertTokenizer, BertModel
import torch

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Function to extract BERT embeddings
def extract_features(texts):
    inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embedding

# from sklearn.feature_extraction.text import TfidfVectorizer

# # Feature Extraction using TF-IDF
# def extract_features(X_train, X_test):
#     vectorizer = TfidfVectorizer(max_features=5000)
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)
#     return X_train_tfidf, X_test_tfidf, vectorizer
