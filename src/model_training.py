import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

# Load tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Define BERT-based Classifier
class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768, 2)  # Output layer for binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0, :])  # CLS token

# Training function
def train_bert_model(model, train_dataloader, optimizer, criterion, device):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# import pickle

# # Train Model and Save
# def train_and_save_model(model, X_train, y_train, model_path):
#     model.fit(X_train, y_train)
#     with open(model_path, 'wb') as file:
#         pickle.dump(model, file)

# # Evaluate Model
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
