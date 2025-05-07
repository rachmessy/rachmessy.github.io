---
layout: default
title: "Reproducibility"
permalink: /reproducibility/
---

# Reproducibility

__LSTM Models__  

- Launch Jupyter Notebook  

- Import all necessary dependencies and set device to MPS   

- Then install wordcloud for the the wordcloud visualization. 
__optimized LSTM model__  
In additon to the above
- Also set up glove url, zip_path, folder and file. Finally, save the model path, the vocab.pickle file  and the embedding matrix for the model fusion 

__BERT Model:__  
- Launch cloud setup of Google Colab pro using GPU for faster computation resources(RAM) and time 

- Ensure to have 4.17 version of colab or higher  

- Due to run time, only 5 epochs was executed  

- same as LSTM model, save and download the vocab.pickle and and the embedding matrix for the model fusion 

__CNN, ResNet50, MultiModal Models__

In order to run all these models, you can download the entire project.
- then you can run this command to create a conda environment suitable for the setup.

{% highlight python %}
conda env create -f environment.yml
{% endhighlight %}

- after you have the environment ready to go, you can just click away in the src/main.ipynb notebook!

