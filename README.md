# Final-Year-Project-Visual-Storytelling-

This project implements and evaluates four different image captioning models: BLIP, CLIP, ViT-GPT2, and CNN-LSTM. The models are evaluated based on the following metrics: 
BLEU-1
BLEU-4
CIDEr
SPICE

The main aim of the project is to create a web app that is targeted towards blind people. The web app has a caption to speech output and visual question answering (VQA) capability in addition to the model comparison.

## Setup

To run the project, you will need to install the following dependencies:

   1. Clone the repository: https://github.com/CyrusFerns233/Final-Year-Project-Visual-Storytelling-
   2. run the demo.ipynb placed in the BLIP folder

You will also need to download the following data: 

    MSCOCO Dataset
    Flickr8k Dataset
    Flickr30k Daatset

Once you have installed the dependencies and downloaded the data, you can run the project by running the following command:

css

   In the colab enviroment open the demo.ipynb and run it.

## BLIP
BLIP (Bootstrapping Language-Image Pre-training) is a state-of-the-art image captioning model that leverages a bootstrapping technique to pre-train a single model for both vision-language understanding and generation tasks. The model achieves this by jointly training on a large corpus of image-caption pairs using a multi-task objective that involves both contrastive and reconstruction losses.The BLIP model is based on a transformer-based neural network architecture, similar to other recent image captioning models. However, unlike other models that pre-train the visual and textual components separately, BLIP uses a bootstrapping approach that enables the model to learn to associate images with their corresponding textual descriptions in a unified framework.The bootstrapping approach involves two stages. In the first stage, the model is pre-trained on a large corpus of text data using a transformer-based language model. The pre-trained language model is then used to initialize the language encoder of the BLIP model.In the second stage, the model is fine-tuned on a large corpus of image-caption pairs using a multi-task objective that involves both contrastive and reconstruction losses. The contrastive loss is used to maximize the similarity between a given image and its corresponding caption, while the reconstruction loss is used to reconstruct the original input image from the generated caption. During training, the model is fed pairs of images and their corresponding captions and is trained to jointly optimize the contrastive and reconstruction losses. This enables the model to learn to associate each image with its correct caption while also learning to generate accurate and diverse captions for new images. Overall, the BLIP model is a powerful and flexible approach to generating natural language descriptions for images that achieves state-of-the-art performance on several benchmark datasets. The model's ability to unify vision-language understanding and generation tasks in a single framework makes it a promising approach for a wide range of vision-language applications.
## CLIP
CLIP is a state-of-the-art image captioning model that is based on a transformer-based neural network architecture. The model has been pre-trained on a large corpus of text and image data, allowing it to learn to associate images with their corresponding textual descriptions.The CLIP model works by encoding both the image and text inputs into a shared feature space, where they are then compared and matched to generate the image caption. This is achieved using a contrastive learning approach, where the model is trained to maximize the similarity between a given image and its corresponding caption, while minimizing the similarity between the image and all other captions in the dataset.To accomplish this, the model first encodes the image and text inputs into feature vectors using separate transformer networks. The image is processed through a convolutional neural network (CNN), while the text is processed through a transformer-based language model. These feature vectors are then projected into a shared embedding space using a linear projection layer, where they can be compared using a similarity metric such as cosine similarity.During training, the model is fed pairs of images and their corresponding captions and is trained to maximize the similarity between them while minimizing the similarity between the image and all other captions in the dataset. This forces the model to learn to associate each image with its correct caption and enables it to generate accurate and diverse captions for new images.Overall, the CLIP image captioning model is a powerful and flexible approach to generating natural language descriptions for images and has been shown to outperform previous state-of-the-art methods on several benchmark datasets.
## ViT-GPT2
ViT-GPT2 is a state-of-the-art image captioning model that leverages the power of two popular transformer-based models - Vision Transformer (ViT) and Generative Pre-trained Transformer 2 (GPT2). The model works in two stages. In the first stage, the ViT component processes the input image and generates a feature map that encodes visual information. This feature map is then fed into the GPT2 component in the second stage, which generates a textual description of the image.The ViT component of the model uses a self-attention mechanism to extract relevant visual features from the input image. It divides the image into a grid of patches and then flattens them into a sequence of vectors that can be processed by the transformer network. This allows the model to learn spatial relationships between different regions of the image and extract high-level features such as objects, shapes, and colors.The GPT2 component of the model is responsible for generating natural language descriptions of the input image. It is a powerful language model that has been pre-trained on a large corpus of text data and can generate coherent and diverse text. The model is fine-tuned on an image captioning task to learn the mapping between visual features and textual descriptions.Overall, the ViT-GPT2 model is capable of generating high-quality captions for a wide range of images and can be used in various applications, including image search, content creation, and assistive technology.
## CNN-LSTM
The CNN-LSTM model for image captioning is a deep learning architecture that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to generate captions for images. CNNs are used for feature extraction from images, while LSTMs are used to generate captions based on the features extracted by the CNN.

In this model, the CNN is typically pre-trained on a large dataset such as ImageNet for image classification. The CNN is then fine-tuned on the specific task of image feature extraction for image captioning. The output of the CNN is a set of feature vectors that represent the visual content of the image.

The LSTM network is then used to generate captions based on the feature vectors. The LSTM takes the feature vectors as input and generates a sequence of words that describe the image. At each time step, the LSTM takes the feature vector and the previous word as input and generates a probability distribution over the next word in the sequence. The word with the highest probability is chosen as the next word, and this process is repeated until an end-of-sequence token is generated.

During training, the model is optimized to generate captions that are similar to the ground-truth captions using evaluation metrics such as BLEU and CIDEr. The model is typically trained end-to-end using backpropagation and stochastic gradient descent.

Overall, the CNN-LSTM model for image captioning has achieved state-of-the-art performance on various benchmark datasets and has shown promising results for generating accurate and natural language descriptions of images.
## Evaluation Metrics

Evaluation metrics are used to measure the quality of a machine learning model's output. In the context of natural language processing, evaluation metrics are used to evaluate the quality of machine-generated text against a reference or human-generated text.

The models are evaluated based on the following metrics:

   BLEU:BLEU (Bilingual Evaluation Understudy) is a metric used to evaluate the quality of machine-translated text against human-generated translations. The metric is based on n-gram precision, which measures the number of overlapping n-grams (sequences of n words) between the machine-generated and human-generated text. BLEU scores range from 0 to 1, with higher scores indicating better performance. BLEU is often used for machine translation and image captioning tasks. BLEU-1, BLEU-2, BLEU-3, and BLEU-4 refer to the n-gram precision at different n-gram orders.
    CIDEr: CIDEr (Consensus-based Image Description Evaluation) is a metric used to evaluate the quality of image captions generated by a machine learning model. CIDEr is based on a consensus measure that compares the similarity between a set of reference captions and the generated caption. CIDEr is often used in image captioning tasks and is considered to be a more robust evaluation metric than BLEU.
    SPICE: SPICE (Semantic Propositional Image Caption Evaluation) is a metric used to evaluate the semantic content of image captions generated by a machine learning model. SPICE evaluates the semantic similarity between a generated caption and a set of reference captions based on a set of semantic propositions. SPICE is often used in image captioning tasks and is considered to be a more comprehensive evaluation metric than BLEU and CIDEr.

## Results
![image](https://user-images.githubusercontent.com/76406095/235854593-391a292b-cab1-41c3-80ea-2fb35f3c32fc.png)

The results of the evaluation are as follows:
Model	    BLEU-1	     BLEU-4	      CIDEr       SPICE
BLIP	    -	         44.6	      142.8        29.5
ViT-GPT2	0.771        0.291        1.118         -
CLIP	    85.6         58.3         122.5        24.3
CNN-LSTM	0.50         0.29         0.972         -

Based on the evaluation metrics, the BLIP model performed the best.

## Web App

The web app has the following capabilities:
Caption to Speech Output

The caption to speech output capability allows blind users to hear the image captions read out loud. The user can input an image and the web app will generate a caption using the selected model, and then read out the caption using a text-to-speech engine.
Visual Question Answering (VQA)

The VQA capability allows blind users to ask questions about an image and get an answer. The user can input an image and a question, and the web app will generate an answer using the selected model.
Usage

To use the web app, run the following command:

In the colab enviroment open the demo.ipynb and run it.

This will launch a local web server that you can access in your web browser.
## Conclusion

In this project, we implemented and evaluated four different image captioning models. Based on the evaluation metrics, the BLIP model performed the best. Additionally, we created a web app targeted towards blind people that has a caption to speech output and VQA capability in addition to the model comparison.
