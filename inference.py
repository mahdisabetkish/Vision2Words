import torch.nn.functional as F
import matplotlib.pyplot as plt

ids, image, token, caption = next(iter(loader_flicker))
plt.imshow(image[0].permute(1, 2, 0)/255)

predicted_caption = []
len_caption = 10
model.eval()
start_token = (torch.ones(40)*59).unsqueeze_(0).int()
for _ in range(len_caption):
    with torch.no_grad():
        logit, emb_token = model(image[0].unsqueeze_(0), start_token)
        MAX = F.cosine_similarity(logit[0],model.embed(torch.tensor(data.w2i['boy'])), dim = 0)
        for word in data.vocab:
            input1 = logit[0]
            input2 = model.embed(torch.tensor(data.w2i[word]))
            # print(input1.shape, input2.shape)
            new_similarity = F.cosine_similarity(input1, input2, dim = 0)
            if new_similarity > MAX:
                MAX = new_similarity 
                predicted_word = word
                # print(new_similarity)
                # print(predicted_word)
    next_token = data.w2i[predicted_word]
    next_token = torch.tensor([next_token])
    start_token = torch.cat((start_token[:, 1:], next_token.unsqueeze_(0)), dim = 1)
    predicted_caption.append(predicted_word)
    # print(start_token)
predicted_caption
