---
layout: default
title: "Methodology"
permalink: /methodology/
---

# Methdology

# Text Processing
The two primary models for the project text-based architecture are LSTM and BERT models  
  
Methodology and Technical approach:  

Rationale for LSTM Model: 

LSTM are usually computed to pattern the long-term dependencies in a text model while handling their sequencies. And because you can control tokenization, vocabulary size(vocab) and when further embedded(Glove as used in this project) gives flexibility. Lastly, it is a whole lot less computational in terms of time and resources compared to transformer models.
 
Methodology for LSTM: 

__A – Baseline LSTM Model__ 

1) Set Device to MPS – Moving the tensors and model to MPS to avoid confusion during the model training/evaluation  at where the Datas are stored 

2)Tokenizer and Vocab -  

3)Data Setup and split into train and test datasets  

- Set up the MultilabelBinarizer 
- Encode the genre outputs from (a) 
- Using Dataloader, setup a batch_size of 32 

4)Model Definition – This is baseline LSTM Model, with only 1 layer, no dropout and data balancing applied not to upset the underrepresented genres in the dataset 

5)Then training and evaluation loop 

__B – Optimized LSTM Model (Using Glove Embeddings)__ 

- Download, extract and loaded 400000 words vector from glove   

- Follow same steps as baseline LSTM model in i) - iii) above, however layers are increased to 2, batch_size stays at 32 and dropout was concluded at 0.5 

- Embed Glove words to set up the embedding_matrix  

- Using an optimal threshold 

- Evaluate the model – training and testing loop 

__Bert Model__

We also wanted to see how a pretrained text model would perform against the LSTM.

Rationale for BERT Model: 

BERT outperforms the LSTM model on NLP classification tasks. Because the BERT model is pretrained, the transfer learning process gives good model performance. Lastly, the high evaluation metrics associated with the BERT model.
 
BERT Model Methodology 

1)Due to complex computation involved with BERT model, it was done using a cloud provider. 

2) Load the dataset, apply the multibinarizer function to encode and transform the genre columns, and create a dataframe to match both the genre and the newly encoding 

3)Then prepare the dataset elements like the tokenizer ahead of the transfer learning using the pretrained BERT model 

4)Finally, we define our tokenizer and model from BERT pre-trained model 

5)Then we evaluate the model, train and test the model 

(Run time was 2hrs 31mins) 

# Image Processing 

To explore our image processing options we primariliy looked at two architectures: a simple CNN model and the pretrained ResNet50 model.

The rationale behind trying the CNN model was mainly to see what type of baseline performance we could achieve with a relatively simple implementation. CNNs are commonly used in vision tasks due to their ability to pick out and learn visual features. I believe in solving problems in the simplest way possible - therefore, the CNN was the natural place to begin before jumping into any of the fancy pretrained models.

The rationale behind ResNet-50 is simply because the CNN didn't perform as well as we hoped. the ResNet is trained on millions of images and has a headstart on detecting features compared to our CNN which hadn't been trained before. ResNet-50 is also a benchmark model and has proven time and time again to perform well on many different vision tasks.

__CNN Model__

The layers:

Conv2d(3, 32, kernel_size=3, padding=1)

BatchNorm2d(32)

ReLU

MaxPool2d(2)

- these layers are stacked 4 times with an increasing filter size
- then we apply a global average pooling to reduce the spatial dimension to 1x1
- then we pass it thorugh a dropout layer and a linear classifying layer
- finally we use sigmoid activation to output the multi label probabilities

complete CNN structure:
{% highlight python %}
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)           
        )

    def forward(self, x):
        x = self.features(x)                      
        x = self.gap(x).flatten(1)               
        x = self.classifier(x)                     
        return torch.sigmoid(x)  
{% endhighlight %}

- before passing the images into the model we apply the following transformations: 
{% highlight python %}
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
{% endhighlight %}

- loss function used:
{% highlight python %}
criterion = nn.BCELoss()
{% endhighlight %}

- optimizer used:
{% highlight python %}
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
{% endhighlight %}

__ResNet-50 Model__

As we mentioned before, the CNN model did not perform as well as we were hoping (as seen in the results section). Therefore we resorted to using a ResNet-50 pretrained model.

- again our base model is the ResNet-50 but we modify it as follows:
{% highlight python %}
class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5, freeze_backbone=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad_(False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)
{% endhighlight %}

- we replace the fully connected head with:
{% highlight python %}
self.backbone.fc = nn.Sequential(
    nn.Linear(in_feats, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
    nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
    nn.Linear(128, num_classes)
)
{% endhighlight %}

- the parameters that we perform a grid search on are (see results page): 
{% highlight python %}
param_grid = {
    "dropout_p":    [0.3, 0.5],
    "lr":           [1e-4, 3e-4],
    "weight_decay": [1e-4, 1e-5],
    "freeze":       [True, False]
}
{% endhighlight %}

# Fusion Technique

We concatenated our best models (Bert and ResNet50) models to form the multi-modal classifier
we use our Bert classifier as the backbone for the text.
we use our ResNet50 model as the backbone for the images.
we also define a projection to fuse the two models:
{% highlight python %}
self.proj = nn.Sequential(
    nn.Linear(num_classes, 128),
    nn.ReLU(),
    nn.Dropout(0.3)
)
{% endhighlight %} 

the project text and image features are then concatenated:
{% highlight python %}
combined = torch.cat((text_feat, image_feat), dim=1)
{% endhighlight %}

the final classifier is:
{% highlight python %}
self.classifier = nn.Linear(256, num_classes)
{% endhighlight %}

put together it looks like this:
{% highlight python %}
class MultiModalClassifier2(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model

        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.image_model.parameters():
            p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, image_input):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_logits = text_output.logits

        image_logits = self.image_model(image_input)

        text_feat = self.text_proj(text_logits)
        image_feat = self.image_proj(image_logits)

        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.classifier(combined)
{% endhighlight %}

- the loss function used was BCEWithLogitsLoss()
- the optimizer we used was Adam, with a lr=1e-3
~
~
