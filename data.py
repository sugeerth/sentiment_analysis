import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(path):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def preprocess(texts):
    preprocessed_texts = [text.lower() for text in texts]
    return preprocessed_texts

def tokenize(texts, tokenizer):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return inputs

# Load and split the dataset
train_texts, train_labels = load_dataset('data.csv')
test_texts, test_labels = train_texts[-20:], train_labels[-20:]
train_texts, train_labels = train_texts[:-20], train_labels[:-20]

# Preprocess and tokenize the training and testing data
preprocessed_train_texts = preprocess(train_texts)
preprocessed_test_texts = preprocess(test_texts)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_inputs = tokenize(preprocessed_train_texts, tokenizer)
test_inputs = tokenize(preprocessed_test_texts, tokenizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

train_inputs = {k: v.to(device) for k, v in train_inputs.items()}
train_labels = torch.tensor(train_labels).to(device)

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(**train_inputs, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
test_labels = torch.tensor(test_labels).to(device)

model.eval()
with torch.no_grad():
    outputs = model(**test_inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    labels = test_labels.cpu().numpy()

accuracy = accuracy_score(labels, predictions)
cm = confusion_matrix(labels, predictions)

print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
