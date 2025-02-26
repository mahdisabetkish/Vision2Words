import torch
import torch.nn as nn 
from torchvision import models
import random
from Flicker_Data import Flicker8k_loader, data

class Captioner(nn.Module):
    def __init__(self, vocab_size, embedd_size, hidden_size, output_size, teacher_forcing_ratio=0.5):
        super(Captioner, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.embedding_layer = nn.Embedding(vocab_size, embedd_size)
        
        efficientnet_b0 = models.efficientnet_b0(pretrained=True)
        for param in efficientnet_b0.features.parameters():
            param.requires_grad = False
        efficientnet_b0.features.eval()
        
        self.feature_extractor = nn.Sequential(
            efficientnet_b0.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.reducer = nn.Linear(1280, embedd_size)
        self.lstm = nn.LSTM(embedd_size * 2, hidden_size, batch_first=True)  
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, image, token):
        batch_size, num_caption, seq_len = token.size()
        # print('Original:', token.shape, image.shape)
        features = self.feature_extractor(image)
        features = self.reducer(features)
        features = features.unsqueeze(1)
        features = features.expand(-1, num_caption, -1)
        features = features.reshape(batch_size * num_caption, -1)
        token = token.reshape(batch_size * num_caption, -1)
        embed_token = self.embedding_layer(token)
        # print('all_len: ', features.shape, embed_token.shape)
        # print(seq_len)
        features = features.unsqueeze(1)
        h, c = None, None
        all_token = []
        

        for t in range(seq_len):
            if t == 0:
                input_token = embed_token[:, t, :]
            else:
                use_teacher_forcing = random.random() < self.teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = embed_token[:, t, :]
                else:
                    input_token = self.embedding_layer(predicted_token.squeeze(-1))  
                    
            input_token = input_token.unsqueeze(1)
            # print(input_token.shape)
            # print(input_token.shape, features.shape)
            lstm_input = torch.cat([features, input_token], dim = 2)
            # print('lstm', lstm_input.shape)
            output, (h, c) = self.lstm(lstm_input, (h, c) if h is not None else None)
            output = output.squeeze(1)
            logits = self.classifier(output)
            all_token.append(logits)
            probs = torch.softmax(logits, dim = -1)
            predicted_token = torch.multinomial(probs, num_samples = 1)

        all_logits = torch.stack(all_token, dim=1)
        return all_logits
<<<<<<< HEAD
        
=======


>>>>>>> 953beb5 (this is the first push from Ubuntu and it seems the LSTM works fine)
vocab_size = len(data.vocab)
embedd_size = 32
hidden_size = 64
output_size = vocab_size
model = Captioner(vocab_size, embedd_size, hidden_size, output_size)
