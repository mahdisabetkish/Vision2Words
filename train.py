import flicker_loader
import Model
import importlib 

from flicker_loader import Data, loader_flicker


importlib.reload(flicker_loader)
importlib.reload(Model)

caption_file = r'C:\Users\shahmir\Desktop\Flickeer8k\flickr8k\captions.txt'
image_dir = r'C:\Users\shahmir\Desktop\Flickeer8k\flickr8k\Images'

data = Data(caption_file, image_dir)

vocab = data.unique_vocab()
vocab_size = len(vocab)
hidden_size = 128
output_size = 32
embedd_size = 32
model = Model.Engin(vocab_size, hidden_size, output_size, embedd_size)

epochs = 3
learning_rate = 0.001
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(params = model.parameters(), lr = learning_rate)
stepi = []; lossi = []; counter = 0
model.train()
for epoch in (range(epochs)):
    for ids, image, token, caption in tqdm(loader_flicker):
    # ids, image, token, caption = next(iter(loader_flicker))
        for j in range(5):
            optimizer.zero_grad()  
            logits, emb_token = model(image, token[:, j, :])
            total_loss = 0  
            for i in range(40):
                target = torch.ones(token.shape[0])
                loss = criterion(logits, emb_token[:, i, :], target)
                total_loss += loss
            total_loss.backward() 
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.abs().mean()}")
            optimizer.step()  
            stepi.append(counter); lossi.append(total_loss.item()); counter += 1
