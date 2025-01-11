import os
from PIL import Image
import numpy as np 
import albumentations as A
import torch
import string 
import random 

import matplotlib.pyplot as plt 

from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, caption_file, images_dir):
        super(Data, self).__init__()
        self.caption_file = caption_file
        self.images_dir = images_dir
        self.captions, self.ids = self.read_captions_ids()
        self.vocab = self.unique_vocab()
        self.w2i, self.i2w = self.word_index()
        self.token = self.tokenizer()
       

    def read_captions_ids(self):
        with open(self.caption_file, 'r') as f:
            lines = f.readlines()
        captions = {}
        ids = []
        for line in lines[1:]:
            image_id, caption = line.strip().split(',', 1)
            if image_id in captions:
                captions[image_id].append(caption)
            else:
                captions[image_id] = [caption]
                ids.append(image_id)
        # lower case and removing punctuations 
        translator = str.maketrans('', '', string.punctuation)
        for i in range(len(ids)):
            for j in range(5):
                captions[ids[i]][j] = captions[ids[i]][j].lower().translate(translator)
                captions[ids[i]][j] = '<START>' + ' ' +  captions[ids[i]][j] + '<END>'
        
        return captions, ids

    def read_images(self, ix):
        images_path = os.path.join(self.images_dir,self.ids[ix])
        image = Image.open(images_path)
        image = np.array(image)
        transform = A.Resize(height=224, width=224)
        image = transform(image=image)['image']
        image = image.transpose((0, 1, 2))
        image = torch.tensor(image)
        return image

    def unique_vocab(self):
        vocab_set =set()
        for ix in self.ids:
            for count in range(len(self.captions[ix])):
                for word in (self.captions[ix][count].split(' ')):
                    vocab_set.add(word)
        return vocab_set

    def word_index(self):
        special_token = ['<START>', '<END>', '<UNK>', '<PAD>']
        vocab_list = special_token + sorted((self.vocab))
        w2i = {word: idx for idx, word in enumerate(vocab_list)}
        i2w = {word: idx for word, idx in enumerate(vocab_list)}
        return w2i, i2w

    def tokenizer(self):
        #Max Length Token
        Max_len = 0
        for ix in self.ids:
            for sentence in self.captions[ix]:
                sen_len = len(sentence.split(' '))
                if sen_len > Max_len:
                    Max_len = sen_len
        # Tokenizing     
        token = {}
        for ix in self.ids:
            token[ix] = []
            for sentence in self.captions[ix]:
                new_token = [self.w2i[word] for word in sentence.split(' ')] 
                token[ix].append(new_token)
        # Padding
        for ix in self.ids:
            for tok in token[ix]:
                for pad in range(Max_len-len(tok)):
                    tok.append(self.w2i['<PAD>'])
        return token

    def ploter(self, index = None):
        plt.figure(figsize=(10,10))
        for i in range(4):
            if index == None:
                ix = random.randint(0,8091)
            else:
                ix = index[i]
            plt.subplot(2, 2, i+1)
            image= self.read_images(ix)
            plt.imshow(image)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ix):
        token = self.tokenizer()
        return self.ids[ix], self.read_images(ix), token[self.ids[ix]], self.captions[self.ids[ix]]
            

def flicker_collate_fn(batch):
    ids = []
    images = []
    tokens = []
    captions = []
    for ides, image, token, caption in batch:
        images.append(image)
        captions.append(caption)
        ids.append(ides)
        tokens.append(token)
    
    images = torch.stack(images)
    tokens = torch.tensor(tokens)
    return ids, images.permute(0, 3, 1, 2).float(), tokens, captions
    
image_dir = r'C:\Users\shahmir\Desktop\Flickeer8k\flickr8k\Images'
caption_file = r'C:\Users\shahmir\Desktop\Flickeer8k\flickr8k\captions.txt'

data =Data(caption_file, image_dir)
loader_flicker = DataLoader(data, batch_size = 16, shuffle = True, collate_fn = flicker_collate_fn)
      


        
         

