import torch
import torch.nn as nn 

class Image_captioning(nn.Module):
    def __init__(self, input_size, hidden_size, embedd_dim, vocab_size):
        super(Image_captioning, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedd_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)

    def forward(self, feature, token):
        embedded_token = self.embedding(token)
        feature = feature.unsqueeze(1).repeat(1, token.shape[1], 1)  # Shape: [4, 39, 128]
        lstm_input = feature + embedded_token
        out, h = self.lstm(lstm_input)
        return out, embedded_token
        
