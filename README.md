# Contrastive Predictive Coding

### Concept

The package contains Keras implementation of Contrastive Predictive Coding algorithm for audio signal described in [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).
  
Main purpose of the algorithm is to extract high-level i.e. conceptual encoded context features for better description of the signal. The algorithm uses Contrastive Predictive Coding which means:
* __contrastive__: the model learns to distinguish "correct" and "incorrect" sequences, where a "correct" sequence is a sequence sampled from the same data subset (the image, audio recored, category etc) and an "incorrect" sequence is a sequence sampled from another data subset. This approach is analogue of negative sampling in training of Word2Vec embeddings.
* __predictive__: the model learns to predict predict a sequence continuation from a known part of the the same sequence. For instance: predict left part of the image from right part of the image or predict next part of the audio record.
* __coding__: the model works in high-level context representation of the signals. In other words, the model predicts context of the image (i.e. clouds, dogs or water) rather value of each pixel on the image.


### Package structure

### How to

### Usage examples