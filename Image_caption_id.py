import os
from PIL import Image
import numpy as np 
import albumentations as A
import torch
import string 

class Data():
    def __init__(self, caption_file, images_dir):
        super(Data, self).__init__()
        self.caption_file = caption_file
        self.images_dir = images_dir
        self.captions, self.ids = self.read_captions_ids()
        self.vocab = self.unique_vocab()
        self.w2i, self.i2w = self.word_index()

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

            


        
         

