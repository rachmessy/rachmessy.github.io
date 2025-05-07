---
layout: default
title: "Results"
permalink: /results/
---

# Results

__LSTM Results__

__Model A__

<img src="/images/lstm.png" alt="distribution" width="300" height="200">

__Model B__

<img src="/images/lstmopt.png" alt="distribution" width="300" height="200">

<img src="/images/lstmopt1.png" alt="distribution" width="300" height="200">

__Bert Results__

<img src="/images/bert.png" alt="distribution" width="300" height="200">

<img src="/images/bert_2.png" alt="distribution" width="300" height="200">

<img src="/images/bert_3.png" alt="distribution" width="300" height="200">

__SimpleCNN Results__

- although the CNN did have better than random guessing result, the model simply wouldn't learn much above .25 f1 score (even with training for more epochs). 

<img src="/images/cnn_training_results.png" alt="distribution" width="500" height="200">

__ResNet50 Results and Evaluation__

<img src="/images/resnet_dp0.3_lr0.0003_wd0.0001_frzTrue_curves.png" alt="distribution" width="500" height="200">

- here are the individual genre f1 scores. as we can see our best performing genre is Animation and our wors
t performing genre is Mystery.

<img src="/images/resnet2.png" alt="distribution" width="300" height="200">

__Multi-Modal Results and Evaluation__

- although the BERT model on its' own performed well, we unfortunately were unable to get the multimodal approach to perform as good. It seems to have been hindered by the performance of our ResNet50 model.

<img src="/images/multimodal_training_results.png" alt="distribution" width="500" height="200">

- here are the individual genre f1 scores. as we can see our best performing genre is Animation and our worst performing genre is Mystery.

<img src="/images/multimodal2.png" alt="distribution" width="300" height="400">

Across all of our models Animation continued to be the best performing class while Mystery most times was the worst performing class. 
