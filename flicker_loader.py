import os
import random
import string
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

class Data(Dataset):
    """
    PyTorch Dataset for the Flickr8k dataset.

    Attributes:
        caption_file (str): Path to the captions file.
        images_dir (str): Path to the images directory.
        captions (dict): Dictionary mapping image IDs to lists of captions.
        ids (list): List of all image IDs.
        vocab (set): Set of unique words in the captions.
        w2i (dict): Word-to-index mapping.
        i2w (dict): Index-to-word mapping.
        token (dict): Pre-tokenized and padded captions for each image ID.
    """
    def __init__(self, caption_file, images_dir):
        super(Data, self).__init__()
        self.caption_file = caption_file
        self.images_dir = images_dir

        # Load captions and image IDs
        self.captions, self.ids = self._read_captions_ids()

        # Build vocabulary and word-index mappings
        self.vocab = self._build_vocab()
        self.w2i, self.i2w = self._build_word_index()

        # Tokenize captions and apply padding
        self.token = self._tokenizer()

        # Define image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def _read_captions_ids(self):
        """
        Reads captions and image IDs from the captions file.

        Returns:
            captions (dict): Dictionary of image IDs to captions.
            ids (list): List of image IDs.
        """
        with open(self.caption_file, 'r') as f:
            lines = f.readlines()

        captions = {}
        ids = []
        translator = str.maketrans('', '', string.punctuation)

        for line in lines[1:]:  # Skip the header
            image_id, caption = line.strip().split(',', 1)
            caption = caption.lower().translate(translator)
            caption = f"<START> {caption} <END>"

            if image_id in captions:
                captions[image_id].append(caption)
            else:
                captions[image_id] = [caption]
                ids.append(image_id)

        return captions, ids

    def _build_vocab(self):
        """
        Creates a set of unique words from the captions.

        Returns:
            vocab (set): Unique words in the captions.
        """
        vocab = set()
        for captions in self.captions.values():
            for caption in captions:
                vocab.update(caption.split())
        return vocab

    def _build_word_index(self):
        """
        Creates word-to-index and index-to-word mappings.

        Returns:
            w2i (dict): Word-to-index mapping.
            i2w (dict): Index-to-word mapping.
        """
        special_tokens = ['<START>', '<END>', '<UNK>', '<PAD>']
        vocab_list = special_tokens + sorted(self.vocab)

        w2i = {word: idx for idx, word in enumerate(vocab_list)}
        i2w = {idx: word for idx, word in enumerate(vocab_list)}

        return w2i, i2w

    def _tokenizer(self):
        """
        Tokenizes and pads all captions.

        Returns:
            token (dict): Tokenized and padded captions for each image ID.
        """
        max_len = max(len(caption.split()) for captions in self.captions.values() for caption in captions)

        token = {}
        for image_id, captions in self.captions.items():
            token[image_id] = [
                [self.w2i.get(word, self.w2i['<UNK>']) for word in caption.split()] +
                [self.w2i['<PAD>']] * ( max_len - len(caption.split()))
                for caption in captions
            ]

        return token

    def _read_image(self, image_id):
        """
        Reads and preprocesses an image.

        Args:
            image_id (str): Image ID to read.

        Returns:
            image (Tensor): Preprocessed image tensor.
        """
        image_path = os.path.join(self.images_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def plot_sample_images(self, indices=None):
        """
        Plots a random sample of images along with their captions.

        Args:
            indices (list, optional): List of specific indices to plot.
        """
        if indices is None:
            indices = random.sample(range(len(self.ids)), 4)

        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(indices):
            plt.subplot(2, 2, i + 1)
            image = self._read_image(self.ids[idx])
            plt.imshow(image.permute(1, 2, 0))
            plt.title("\n".join(self.captions[self.ids[idx]]))
            plt.axis("off")
        plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single data point.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple: (image_id, image, tokenized captions, raw captions).
        """
        image_id = self.ids[idx]
        image = self._read_image(image_id)
        tokens = self.token[image_id]
        captions = self.captions[image_id]
        return image_id, image, tokens, captions


def flicker_collate_fn(batch):
    """
    Custom collate function to batch data points.

    Args:
        batch (list): List of data points.

    Returns:
        Tuple: (image_ids, images, tokens, captions).
    """
    image_ids, images, tokens, captions = zip(*batch)

    # Convert images to a single tensor
    images = torch.stack(images)

    # Pad tokens to the same length
    tokens = [torch.tensor(t) for t in tokens]
    tokens = torch.stack(tokens)

    return image_ids, images, tokens, captions


# Paths to data
image_dir = r"C:\Users\Apple\Desktop\Flickeer8k\flickr8k\Images"
caption_file = r"C:\Users\Apple\Desktop\Flickeer8k\flickr8k\captions.txt"

# Create dataset and dataloader
data = Data(caption_file, image_dir)
loader_flicker = DataLoader(data, batch_size=2, shuffle=True, collate_fn=flicker_collate_fn)
