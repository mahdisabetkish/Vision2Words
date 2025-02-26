import torch
import string
import os
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Data(Dataset):
    def __init__(self, images_dir, captions_file):
        super(Data, self).__init__()
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.ids, self.captions = self.read_captions()
        self.w2i, self.i2w, self.vocab = self.word_index()
        self.token = self.tokenizer()
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

    def read_captions(self):
        with open(self.captions_file, 'r') as f:
            lines = f.readlines()
        translator = str.maketrans('', '', string.punctuation)
        ids = [] 
        captions = {}
        for line in lines[1:]:
            new_id, new_caption = line.strip().split(',', 1)
            new_caption = '<START>' + ' ' + new_caption.lower().translate(translator) + ' ' + '<END>'
            if new_id in captions:
                captions[new_id].append(new_caption)
            else:
                captions[new_id] = [new_caption]
                ids.append(new_id)
        return ids, captions

    def word_index(self):
        captions = list(self.captions.values())
        original_vocab = set()
        for each_caption in captions:
            for sentence in each_caption:
                original_vocab = original_vocab.union(sentence.split())
        special_tokens = ["<START>", "<END>", "<PAD>"]
        rest_of_vocab = [word for word in original_vocab if word not in special_tokens]
        ordered_vocab = special_tokens + rest_of_vocab
        i2w = {i :j for i,j in enumerate(ordered_vocab)}
        w2i = {j :i for i,j in enumerate(ordered_vocab)}
        return w2i, i2w, ordered_vocab

    def tokenizer(self):
        max_length = max(len(caption.split()) for captions in self.captions.values() for caption in captions)
        token = {}
        for ids, captions in self.captions.items():
            token[ids] = [
            [self.w2i[word] for word in caption.split()] + (max_length-len(caption.split()))*[self.w2i['<PAD>']]
            for caption in captions
            ]
        return token 

    def image_reader(self, image_id):
        image_path = os.path.join(self.images_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ix):
        image_id = self.ids[ix]
        token = self.token[image_id]
        image = self.image_reader(image_id)
        caption = self.captions[image_id]
        return image_id, token, image, caption



def flicker_collate_fn(batch):
    # Unpack the batch
    image_ids, tokens, images, captions = zip(*batch)
    
    # Convert images to a tensor (already preprocessed in Data class)
    images = torch.stack(images, dim=0)
    
    # Convert padded tokens to a tensor
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Return as a dictionary (adjust keys based on your model's input)
    return image_ids, tokens, images, captions

image_dir = '/home/shahmir/Desktop/Vision2words_ubuntu/DATA/Images'
caption_file = '/home/shahmir/Desktop/Vision2words_ubuntu/DATA/captions.txt'
data = Data(image_dir, caption_file)
Flicker8k_loader = DataLoader(data, batch_size = 32, shuffle = True, collate_fn = flicker_collate_fn)
print('Done')
#>>>>>>> 953beb5 (this is the first push from Ubuntu and it seems the LSTM works fine)
