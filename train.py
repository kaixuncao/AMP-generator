import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from tokenizer.simple_tokenizer import SimpleTokenizer

class SequenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        encoded_sequence = self.tokenizer.encode(sequence, add_special_tokens=True)
        if len(encoded_sequence) > self.max_length:
            encoded_sequence = encoded_sequence[:self.max_length]
        elif len(encoded_sequence) < self.max_length:
            encoded_sequence.extend([self.tokenizer.vocab['[PAD]']] * (self.max_length - len(encoded_sequence)))
        return torch.tensor(encoded_sequence, dtype=torch.long)

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data = pd.read_pickle("processed_data.pkl")

    # 加载 tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # 定义数据集和数据加载器
    dataset = SequenceDataset(data, tokenizer, max_length=129)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 定义模型、损失函数和优化器
    vocab_size = tokenizer.vocab_size
    embedding_dim = 128
    hidden_dim = 256
    model = SimpleModel(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['[PAD]'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 15
    for epoch in range(num_epochs):
        epoch_loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    main()
