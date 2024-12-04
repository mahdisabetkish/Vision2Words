
# Flickr8k Image Captioning Project
🚧This repository is currently being developed.

Welcome to the **Flickr8k Image Captioning** project! This repository contains code for training a model to generate captions for images from the **Flickr8k dataset**. The goal is to build an AI system that can interpret images and generate natural language descriptions.



## Project Overview

The **Flickr8k dataset** consists of 8,091 images with corresponding captions. Each image in the dataset is annotated with multiple captions describing the content. This project aims to use these images and captions to train an image captioning model.



<!-- Banner Image Section -->
<!-- Using HTML to resize the image -->
<div style="text-align: center;">
  <img src="https://github.com/mahdisabetkish/Flickr-8k/raw/main/Images/images%20(1).jpg?raw=true" 
       alt="Flickr8k Banner" 
       style="width: 60%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>








## Features

- **Image Captioning**: Generate human-readable captions for images using deep learning models.
- **Pretrained Models**: Leverage models trained on the Flickr8k dataset for efficient caption generation.
- **Easy Setup**: Quick and easy to start with well-documented instructions.

## Dataset
* The dataset is available on [Flicker8K](https://www.kaggle.com/datasets/nunenuh/flickr8k)


## How it Works

This project uses a combination of **Convolutional Neural Networks (CNNs)** for image feature extraction and **Recurrent Neural Networks (RNNs)** or **Transformers** for generating captions. The process can be broken down into the following steps:

1. **Extract Features**: Use a CNN (e.g., ResNet) to extract features from each image in the dataset.
2. **Train the Model**: Train a sequence generation model (e.g., LSTM) on the captions.
3. **Generate Captions**: Use the trained model to generate captions for new images.



## Example Output

Here's an example of an image from the **Flickr8k dataset** with a generated caption:

![](https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/image01.jpg)

*Generated Caption*:A motorcycle is cruising through the heart of the desert.



## Getting Started

You can access Flicker8K dataset using:

```bash
!wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
!unzip -q flickr8k.zip -d ./flickr8k
!rm flickr8k.zip
!echo "Downloaded Flickr8k dataset successfully."
```

To get started with this project, clone the repository and follow the installation instructions:


```bash
git clone https://github.com/mahdisabetkish/flickr-8k.git
cd flickr8k-captioning
```
## VGG11 part 

* Feature Extraction with VGG11:

The pretrained VGG11 model is used as a feature extractor to encode images into high-dimensional feature vectors.
The convolutional layers of VGG11 extract spatial and semantic features, which are then flattened into a fixed-size vector.

<div style="text-align: center;">
  <img src="https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/VGG11.png?raw=true" 
       alt="Flickr8k Banner" 
       style="width: 75%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>


## LSTM part 
The Long Short-Term Memory (LSTM) network in the context of an image captioning system plays the critical role of converting image features (from the CNN) and textual inputs (captions) into meaningful sequences of words.
The LSTM consists of:

* Cell State: A memory that carries information through the sequence.
* Gates: Three key gates control the flow of information:
* Forget Gate: Decides what information to discard.
* Input Gate: Determines what new information to store.
* Output Gate: Controls the output based on the current cell state.
Each gate uses sigmoid and tanh activation functions to regulate information flow.

<div style="text-align: center;">
  <img src="https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/recurrent.png?raw=true" 
       alt="Flickr8k Banner" 
       style="width: 80%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>
