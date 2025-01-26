
# Image Captioning with EfficientNet-B0 and LSTM

Welcome to the repository for my **Image Captioning** project! This project leverages the power of **EfficientNet-B0** for image feature extraction and an **LSTM** (Long Short-Term Memory) network for generating descriptive captions. Below, I'll walk you through the project, its components, and how you can get started.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Installation](#installation)
4. [Details](#details)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Image captioning is a fascinating area of research that combines **Computer Vision** and **Natural Language Processing (NLP)**. The goal is to generate a textual description of an image, which can be useful in various applications like assistive technologies, image indexing, and more.

In this project, I used **EfficientNet-B0** as the backbone for extracting features from images and an **LSTM** network to generate captions based on those features. The model was trained on the **Flickr8k** dataset, which contains 8,091 images, each paired with five different captions.

## Project Overview

The project is divided into the following key components:

1. **Data Preprocessing**: The images are resized and normalized, and the captions are tokenized and padded.
2. **Feature Extraction**: EfficientNet-B0 is used to extract features from the images.
3. **Caption Generation**: An LSTM network is trained to generate captions based on the extracted features.
4. **Model Training**: The model is trained using a combination of image features and captions.
5. **Inference**: The trained model is used to generate captions for new images.

.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mahdisabetkish/Vision2Words.git
   ```
2. **install packages**
   ```bash
   pip install -r requirements.txt
   ```


# LSTM Neural Network

**Long Short-Term Memory** (LSTM) is a type of recurrent neural network (RNN) designed to handle sequential data, such as time series, text, or speech. Unlike traditional RNNs, LSTMs excel at capturing long-term dependencies by addressing the problem of vanishing gradients during training

<div align="center">
  <img src="https://github.com/mahdisabetkish/Vision2Words/blob/main/Images/LSTMGIF.gif" alt="LSTM Neural Network" width="500px" style="border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
</div>

## Details
### Custom DataLoader: How It Works

This custom DataLoader is designed for handling **image-caption pairs**, making it ideal for datasets like Flickr8k. It processes the data efficiently and prepares it for training deep learning models, such as those used for **image captioning tasks**.

#### Key Features

1. **Data Reading and Preprocessing**:
   - Reads captions from a `.txt` file and pairs them with their corresponding image IDs.
   - Preprocesses captions by:
     - Converting to lowercase.
     - Removing punctuation.
     - Adding special tokens (`<START>`, `<END>`).

2. **Image Processing**:
   - Resizes images to **224x224 pixels** using the `Albumentations` library.
   - Converts images to tensors for compatibility with PyTorch models.

3. **Vocabulary Creation**:
   - Builds a unique vocabulary from all captions.
   - Creates mappings for:
     - **Word-to-index (`w2i`)** for tokenization.
     - **Index-to-word (`i2w`)** for decoding.

4. **Caption Tokenization and Padding**:
   - Tokenizes captions using the `w2i` mapping.
   - Pads all sentences to the same length with a `<PAD>` token for efficient batching.

5. **Batch Collation**:
   - Custom `collate_fn` ensures:
     - Images are stacked into a tensor of shape `(batch_size, 3, 224, 224)`.
     - Captions are tokenized and padded for model training.
     - Metadata like image IDs and raw captions are preserved in the batch.


#### How to Use

1. **Initialize the Dataset**:
   ```python
   data = Data(caption_file, image_dir)

