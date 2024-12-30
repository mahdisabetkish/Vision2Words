
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
## VGG11

* Feature Extraction with VGG11:

The pretrained VGG11 model is used as a feature extractor to encode images into high-dimensional feature vectors.
The convolutional layers of VGG11 extract spatial and semantic features, which are then flattened into a fixed-size vector.

<div style="text-align: center;">
  <img src="https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/VGG11.png?raw=true" 
       alt="Flickr8k Banner" 
       style="width: 50%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>

## RNN 
**1. Input Size (Input Dimension):**
   
* This parameter refers to the number of features in each input at every time step. In text processing, for example, the input size could correspond to the size of the vocabulary, with each input being a one-hot encoded vector representing a character or word.

* Example: If you are working with a vocabulary of 10 unique characters, the input size will be 10 (i.e., a one-hot vector of length 10 for each character).

**2. Hidden Size (Hidden State Dimension):**
   
* This is the size of the hidden state vector, which stores the internal memory of the RNN. It determines how much information the network can remember about previous time steps. A larger hidden size means the network has more capacity to capture long-term dependencies, but it may also increase the risk of overfitting.

* Example: If the hidden size is set to 50, then at each time step, the network maintains a hidden state vector of length 50.

**3. Output Size (Output Dimension):**
   
* The output size corresponds to the dimension of the output vector at each time step. For many RNNs, the output size is the same as the input size, especially when the task involves predicting the next token in a sequence, such as in character-level language models. However, for tasks like classification, this could be different (e.g., the number of classes).

* Example: In a language model, the output size might be the number of characters in the vocabulary, so if there are 10 unique characters, the output size will also be 10.

**4. Weights (Weight Matrices):**
   
RNNs use weights to transform inputs, hidden states, and outputs at each time step. These include:

* Weight Matrix (Wxh): This matrix connects the input at each time step to the hidden state.
* Weight Matrix (Whh): This matrix connects the hidden state from the previous time step to the current hidden state. It's responsible for passing the "memory" or information between steps.
* Weight Matrix (Why): This matrix connects the hidden state to the output. It helps the network generate predictions based on the current hidden state.

**5. Biases:**
  
* Bias for Hidden State (bh): A bias vector is added to the hidden state before applying an activation function.
* Bias for Output (by): A bias vector is added to the output before applying the final activation function (such as softmax for classification tasks).

  
<div style="text-align: center;">
  <img src="https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/rnn_1.png?raw=true" 
       alt="Flickr8k Banner" 
       style="width: 45%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>

## Multi-layer RNN

* In a multi-layer RNN, each layer receives the output from the previous layer as its input. The first layer processes the raw input sequence, and each subsequent layer refines the representation of the sequence.
* Each layer's hidden state gets passed to the next layer along with its own state from the previous time step.

<div style="text-align: center;">
  <img src="https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/Multi-layer-RNN.png?raw=true" 
       alt="Flickr8k Banner" 
       style="width: 45%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>

## Teacher forcing 

In a traditional RNN, the hidden state at time $(t)$ is computed based on the previous hidden state $( h_{t-1} )$ and the current input $( x_t )$:

$$ h_t = f(W_{ih} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h) $$

In teacher forcing, instead of using the model's own prediction at time $\( t-1 \)$, we use the actual target $\( y_{t-1} \)$ as the input for the next time step:

$$ h_t = f(W_{ih} \cdot y_{t-1} + W_{hh} \cdot h_{t-1} + b_h) $$

Here, $\( y_{t-1} \)$ is the ground truth from the previous time step, which speeds up training.

![teacher_forcing](https://github.com/mahdisabetkish/Flickr-8k/blob/main/Images/teacher_forcing.png)

## LSTM 
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
       style="width: 60%; height: auto; border-radius: 10px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
</div>

## Put all together
A simpler CNN will be implemented, EfficientNet_b0
Calculating the loss using CosinEmbeddingLoss
