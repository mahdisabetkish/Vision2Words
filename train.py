from Flicker_Data import Flicker8k_loader, data
from model import model
from tqdm import tqdm

import torch 
import torch.nn as nn 
import torch.optim as optim
num_epoch = int(input('num_epoch =' ))

vocab_size = len(data.vocab)  
criterion = nn.CrossEntropyLoss(ignore_index = 2)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
stepi = []; lossi=[]; counter = 0
for epoch in tqdm(range(num_epoch)):
    ix, token, image, caption = next(iter(Flicker8k_loader))
    optimizer.zero_grad()
    all_logits = model(image, token)
    all_logits = all_logits.view(-1, vocab_size)  
    targets = token.view(-1) 
    
    loss = criterion(all_logits, targets) 
    loss.backward()  
    optimizer.step()  
    counter += 1; lossi.append(loss.item()); stepi.append(counter)

