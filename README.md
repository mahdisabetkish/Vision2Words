
# Image Captioning with EfficientNet-B0 and LSTM

Welcome to the repository for my **Image Captioning** project! This project leverages the power of **EfficientNet-B0** for image feature extraction and an **LSTM** (Long Short-Term Memory) network for generating descriptive captions. Below, I'll walk you through the project, its components, and how you can get started.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Installation](#installation)
4. [EfficientNet](#EfficientNet)
5. [LSTM](#LSTM)
6. [Details](#details)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

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
   
## EfficientNet-b0
EfficientNet-B0 is a convolutional neural network (CNN) designed for high accuracy and computational efficiency. It introduces a novel **compound scaling method**, which balances the model's depth (number of layers), width (number of channels), and resolution (input image size) to achieve better performance with fewer resources.

### Key Features:
1. **Compound Scaling**:
   - Instead of arbitrarily scaling one dimension (e.g., depth or width), EfficientNet scales all three dimensions simultaneously using a fixed scaling coefficient.

2. **Architecture**:
   - The network starts with an **initial convolutional layer** for feature extraction.
   - It uses a series of **mobile inverted bottleneck convolution (MBConv)** layers, which combine depthwise separable convolutions and efficient skip connections.
   - Each block extracts hierarchical features, refining details as the input progresses through the network.
   - The final layers include **global average pooling** and a **fully connected classification head**.

3. **Efficiency**:
   - EfficientNet-B0 uses **fewer parameters** and FLOPs compared to traditional CNNs like ResNet or VGG while maintaining state-of-the-art accuracy.
   - It performs well on a wide range of computer vision tasks, such as image classification, object detection, and segmentation.

EfficientNet-B0’s design demonstrates that thoughtful scaling and architecture optimization can lead to better trade-offs between performance and computational cost.

## LSTM Neural Network
**Long Short-Term Memory** (LSTM) is a type of recurrent neural network (RNN) designed to handle sequential data, such as time series, text, or speech. Unlike traditional RNNs, LSTMs excel at capturing long-term dependencies by addressing the problem of vanishing gradients during training

<div align="center">
  <img src="https://github.com/mahdisabetkish/Vision2Words/blob/main/Images/LSTMGIF.gif" alt="LSTM Neural Network" width="600px" style="border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
</div>

## Details

### Custom DataLoader

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
   loader_flicker = DataLoader(data, batch_size=32, shuffle=True, collate_fn=flicker_collate_fn)

### `Engin` Model

The `Engin` model combines the power of **EfficientNet-B0** for feature extraction with an **LSTM-based sequence model** to process both image features and tokenized text data. Below is a breakdown of how the model works:

#### Key Components:
1. **EfficientNet-B0 Feature Extractor**:
   - The model uses a pre-trained **EfficientNet-B0** as the backbone for image feature extraction.
   - The last layer of the EfficientNet is removed (`[:-1]`), and a **flattening layer** is added to produce a feature vector of size `1280` for each image.
   - The parameters of the EfficientNet feature extractor are **frozen** to avoid updating them during training, making the model more efficient.

2. **Embedding Layer**:
   - The text input (tokens) is passed through an **embedding layer** of size `(vocab_size + 4, embedd_size)` with a padding index of `3`.
   - This layer converts token indices into dense vectors of size `embedd_size`.

3. **LSTM Network**:
   - A **2-layer LSTM** processes the combined features of the image and tokens. 
   - The input to the LSTM is a combination of the image feature vector (`1280`) and the embedded token vectors, reshaped to match the token's batch dimensions.

4. **Fully Connected Layer**:
   - The output of the LSTM is passed through a **fully connected layer** to map the hidden states to the embedding size (`embedd_size`).

5. **Forward Pass**:
   - **Image Input**: The image is processed by the EfficientNet feature extractor to generate a feature vector.
   - **Token Input**: Tokens are embedded using the embedding layer.
   - **Combining Features**: The image features are added to the reshaped token embeddings.
   - **Sequence Modeling**: The combined features are passed through the LSTM to capture sequential dependencies.
   - **Final Output**: The LSTM outputs are transformed into logits through the fully connected layer, and the embedded tokens are returned for further use.

#### Summary:
The `Engin` model is designed to efficiently process **image and text inputs** in a unified framework. It extracts high-level image features using a pre-trained EfficientNet-B0 and combines them with token embeddings, which are then processed by an LSTM to capture sequential relationships. This architecture is useful for tasks such as **image captioning** or **vision-language tasks**.

### Training the Vision-Language Model

This script trains the `Engin` model on the **Flickr8k Dataset**, which combines image and caption data to learn a joint representation. Below is an explanation of the key steps in the training process.


#### Key Components:

1. **Data Preparation**:
   - The `Data` class is used to preprocess images and captions.
   - Captions are tokenized, padded, and embedded, while images are resized and passed through a feature extractor.
   - The `loader_flicker` DataLoader is used to create batches of image-caption pairs for training.

2. **Model Initialization**:
   - The `Engin` model from the `Model` module is instantiated.
   - Parameters include:
     - `vocab_size`: Number of unique words in the vocabulary.
     - `hidden_size`: Dimensionality of the LSTM's hidden state.
     - `output_size` and `embedd_size`: Sizes for output and embeddings.

3. **Training Setup**:
   - **Loss Function**: `CosineEmbeddingLoss` is used to minimize the difference in direction (cosine similarity) between predicted and target embeddings.
   - **Optimizer**: Adam optimizer is configured with a learning rate of `0.001`.

4. **Training Loop**:
   - The training runs for the specified number of epochs.
   - In addition to Ids and captions, each batch from the DataLoader includes:
     - `image`: Batch of image tensors.
     - `token`: Tokenized captions, each with five variations (e.g., different descriptions for the same image).
   - For each tokenized caption:
     - Forward pass through the model to generate predicted embeddings (`logits`) and token embeddings.
     - Compute the total loss by iterating over 40 tokens and comparing their embeddings with a target similarity of `1` using the loss function.
   - The loss is backpropagated, and the optimizer updates the model weights.
   - Loss values and iteration steps are logged for analysis.

#### Summary:

This code implements a custom training loop for the `Engin` model to jointly learn representations of images and captions using a **vision-language approach**. The model leverages:
- **EfficientNet-B0** for image feature extraction.
- **LSTM** for sequential caption modeling.
- **Cosine Embedding Loss** to align the embeddings of image features and their corresponding captions.

By iterating over the data and optimizing the model parameters, the network learns to align image features with their textual descriptions.


