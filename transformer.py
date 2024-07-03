import random
from collections import defaultdict
import requests
import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Path to the text-file
file_path = 'input.txt'

# Open the text-file and read the content
with open(file_path, 'r', encoding='utf-8-sig') as file:
    text = file.read()

# Clean the text
start_index = text.find("CHAPTER I")
end_index = text.find("End of Project Gutenberg's Alice's Adventures in Wonderland")
text = text[start_index:end_index]

# Use spaCy to tokenize the text
doc = nlp(text)

# Extract tokens
tokens = [token.text for token in doc if not token.is_space]


############

# Define sequence length
seq_length = 50

# Generate sequences of tokens
sequences = []
next_words = []

for i in range(len(tokens) - seq_length):
    sequences.append(tokens[i:i + seq_length])
    next_words.append(tokens[i + seq_length])

# Convert sequences and next words to integers
unique_tokens = list(set(tokens))
token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

sequences = np.array([[token_to_id[token] for token in seq] for seq in sequences])
next_words = np.array([token_to_id[token] for token in next_words])


#####
# Parameters
vocab_size = len(unique_tokens)
embedding_dim = 100
rnn_units = 128

num_heads = 2  # Number of attention heads
feed_forward_dim = 512  # Dimensionality of the feed-forward layer
num_layers = 1  # Number of transformer layers

class TransformerRNNModel(nn.Module):
    def __init__(self):
        super(TransformerRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, src):
        # src shape: (batch_size, seq_length)
        embedded = self.embedding(src)  # (batch_size, seq_length, embedding_dim)
        transformer_output = self.transformer_encoder(embedded)  # (batch_size, seq_length, embedding_dim)
        rnn_output, _ = self.rnn(transformer_output)  # (batch_size, seq_length, rnn_units)
        output = self.fc(rnn_output[:, -1, :])  # (batch_size, vocab_size)
        return output


learning_rate = 0.005

# Create the model, loss function, and optimizer

model = TransformerRNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


######


# Convert sequences to PyTorch tensors
sequences_tensor = torch.tensor(sequences, dtype=torch.long)

# labels should be a numpy array of the same length as sequences
labels_tensor = torch.tensor(next_words, dtype=torch.long)  # Adjust dtype if necessary

# Create a TensorDataset
train_dataset = TensorDataset(sequences_tensor, labels_tensor)

# Create DataLoader for batch processing
batch_size = 64  # Adjust based on system's capability
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

######
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

###############
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

#############
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)



def generate_text(model, start_text, num_generate, token_to_id, id_to_token, vocab_size, device, temperature=1.0, top_k=50):
    model.eval()
    
    start_tokens = start_text.split()
    start_sequence = [token_to_id.get(token, token_to_id.get('<unk>')) for token in start_tokens]
    input_eval = torch.tensor([start_sequence], dtype=torch.long).to(device)

    text_generated = []

    for _ in range(num_generate):
        predictions = model(input_eval)
        if predictions.dim() == 2:
            logits = predictions
        else:
            logits = predictions[:, -1, :]  # If predictions include the full sequence

        scaled_logits = logits / temperature
        filtered_logits = top_k_logits(scaled_logits, top_k)
        probabilities = F.softmax(filtered_logits, dim=-1)
        predicted_id = torch.multinomial(probabilities, 1).item()

        text_generated.append(predicted_id)
        input_eval = torch.tensor([[predicted_id]], dtype=torch.long, device=device)

    generated_text = ' '.join([id_to_token.get(id, '<unk>') for id in text_generated])
    return generated_text


#############
# Example usage
start_text = "Alice was walking"
num_generate = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generated_text = generate_text(model, start_text, num_generate, token_to_id, id_to_token, vocab_size, device, temperature=0.8, top_k=40)
print(start_text + " " + generated_text)

