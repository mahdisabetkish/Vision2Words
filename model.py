from torchvision import models

class Engin(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, embedd_size):
        super(Engin, self).__init__()
        weight = models.EfficientNet_B0_Weights.DEFAULT
        
        efficientnet = models.efficientnet_b0(weights = weight)
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1],
                                      nn.Flatten(),
                                        )  # (batch_size, 1280)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.embed = nn.Embedding(vocab_size+4, embedd_size, padding_idx = 3)
        self.lstm = nn.LSTM(1280, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, embedd_size)
         
    def forward(self, image, token):
        emb_token = self.embed(token)
        fe = self.feature_extractor(image)
        features_token = fe + emb_token.view(token.shape[0], -1)
        out, h = self.lstm(features_token)
        logits = self.fc(out)
        return logits, emb_token
        

vocab_size = len(vocab)
hidden_size = 128
output_size = 32
embedd_size = 32
model = Engin(vocab_size, hidden_size, output_size, embedd_size)
