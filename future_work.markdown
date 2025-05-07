---
layout: default
title: "Future Work"
permalink: /future_work/
---

# Discussions

__Learnings:__

 
__Text Models:__  

- LSTM model poor performance with  BERT model performing better on the text dataset  
- Careful and multiple tuning needed for LSTM model than BERT model  
- But BERT model slower and requires more computational time    

__Image Models:__  

- Visual features from the posters has an impact on the model learning process  
- Model evaluation metrics was clipped to underfitting range  
- We observe the image models are prone to misclassification when the visual features are either less or in excess 

__Fused Model:__  

- The combined model showed a bit of similarity in terms of the evaluation metrics  
- While both type of model seem to work with each other, it appears very effective with more simple /complex model tuning     

__Project Pitfalls:__  

- Balanced Dataset: with some genre being way over represented as against the under represented genres   
- With text model suggesting one genre and image model suggesting another from same dataset, this had to be looked into  
- All models for both text, image and fused model were computed with consideration to the different computing platforms 

__Future Work__

- debug why we aren't getting better performance on our fused bert/resnet model.~
