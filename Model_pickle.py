from torchvision import models
import torch.nn as nn
import torch

class Enginpickle(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedd_size):
        super(Enginpickle, self).__init__()
        self.embedd_size = embedd_size
        self.reducer = nn.Linear(1280, embedd_size)
        self.embed = nn.Embedding(vocab_size + 4, embedd_size, padding_idx = 3)
        self.lstm = nn.LSTM(embedd_size * 2, hidden_size, batch_first = True, num_layers = 2)
        self.fc = nn.Linear(hidden_size, embedd_size)
         
    def forward(self, features, token):
        emb_token = self.embed(token)
        reduced_features = self.reducer(torch.tensor(features))
        # reduced_features.unsqueeze_(0)
        # print(reduced_features.shape, emb_token.shape)
        feature_expanded = reduced_features.unsqueeze(1).expand(-1, token.shape[1], -1)
        feature_expanded = feature_expanded.unsqueeze(2).expand(-1, -1, 38, -1)
        # print(feature_expanded.shape, emb_token.shape) 
        combined_features = torch.cat([feature_expanded, emb_token], dim = 3)
        # Reshape for LSTM input
        batch_size, num_captions, seq_len, feature_size = combined_features.shape
        combined_features = combined_features.view(batch_size * num_captions, seq_len, feature_size) 
        # print(combined_features.shape)
        out , (hidden_state, cell_state) = self.lstm(combined_features)
        logit = self.fc(out)
        emb_token = emb_token.view(batch_size * num_captions, 38, self.embedd_size)
        return logit, emb_token
