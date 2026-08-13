import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("test_claims_dataset.csv")
X = df['claim_text'].values
y = (df['true_label'] == 'Fake').astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_preds = svm_model.predict(X_test_tfidf)

svm_acc = accuracy_score(y_test, svm_preds)
svm_f1 = f1_score(y_test, svm_preds)
results['SVM'] = {'Accuracy': svm_acc, 'F1-Score': svm_f1}

from collections import Counter
words = [word for text in X_train for word in text.lower().split()]
vocab = {word: i+2 for i, (word, _) in enumerate(Counter(words).most_common(5000))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def text_to_seq(text, max_len=50):
    seq = [vocab.get(w, 1) for w in text.lower().split()]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

X_train_seq = torch.tensor([text_to_seq(t) for t in X_train], dtype=torch.long)
y_train_seq = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_seq = torch.tensor([text_to_seq(t) for t in X_test], dtype=torch.long)
y_test_seq = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=16, shuffle=True)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(hidden)
        return self.sigmoid(out)

lstm_model = BiLSTM(len(vocab)+2, 100, 64)
criterion = nn.BCELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

lstm_model.train()
for epoch in range(5):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = lstm_model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

lstm_model.eval()
with torch.no_grad():
    lstm_out = lstm_model(X_test_seq)
    lstm_preds = (lstm_out >= 0.5).float().numpy().flatten()
    
lstm_acc = accuracy_score(y_test, lstm_preds)
lstm_f1 = f1_score(y_test, lstm_preds)
results['Bi-LSTM'] = {'Accuracy': lstm_acc, 'F1-Score': lstm_f1}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

def encode_texts(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels, dtype=torch.long))
    return dataset

train_dataset = encode_texts(X_train, y_train)
test_dataset = encode_texts(X_test, y_test)

bert_train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
bert_test_loader = DataLoader(test_dataset, batch_size=16)

bert_optimizer = AdamW(bert_model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

bert_model.train()
for epoch in range(3): 
    for batch in bert_train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        bert_optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        bert_optimizer.step()

bert_model.eval()
bert_preds = []
with torch.no_grad():
    for batch in bert_test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        bert_preds.extend(preds)

bert_acc = accuracy_score(y_test, bert_preds)
bert_f1 = f1_score(y_test, bert_preds)
results['Standard BERT'] = {'Accuracy': bert_acc, 'F1-Score': bert_f1}

for model, metrics in results.items():

