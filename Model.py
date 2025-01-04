class Image_captioning(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Image_captioning, self).__init__()
        self.rnn = nn.RNN(1280+39, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, token,extracted_features):
        for i in range(4):
            tok = token[:, i, :]
            out, h = self.rnn(torch.cat([tok, extracted_features],dim = 1).unsqueeze_(1))
            prob = self.fc(out[:, -1, :])
            return prob
            
model = Image_captioning(256, len(vocab))
efficientnet_features = feature.Feature_extractor()
