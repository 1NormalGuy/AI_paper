import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建语料库
text = "这是一个测试语料库。这个语料库用于训练模型。"
words = nltk.word_tokenize(text)

# 创建训练和测试数据
train_data, test_data = train_test_split(words, test_size=0.2, random_state=42)

# 定义Field
TEXT = Field(tokenize=nltk.word_tokenize, lower=True, init_token='<sos>', eos_token='<eos>')

# 创建数据集
train_dataset, test_dataset = TabularDataset.splits(
    path='.', train='train_data.csv', test='test_data.csv', format='csv', fields=[('Text', TEXT)])

# 构建词汇表
TEXT.build_vocab(train_dataset, min_freq=2)

# 创建迭代器
train_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, test_dataset), batch_size=64, device=torch.device('cuda'))

# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# 训练模型
model = LSTMModel(len(TEXT.vocab), 100, 256, len(TEXT.vocab))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.Text).squeeze(1)
        loss = criterion(predictions, batch.Label)
        loss.backward()
        optimizer.step()

# 计算PPL
def calculate_ppl(sentence):
    model.eval()
    tokenized = [tok for tok in nltk.word_tokenize(sentence)]
    numericalized = [TEXT.vocab.stoi[tok] for tok in tokenized]
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    prediction = torch.exp(model(tensor))
    return prediction.mean().item()

# 测试
test_text = "这是一个测试文本。"
print(calculate_ppl(test_text))