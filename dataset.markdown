---
layout: default
title: "Dataset"
permalink: /dataset/
---

# Dataset Description
__Sources of Data__

The movie dataset we used was curated from The Movie Database - TMDb API. We used TMDb's /discover/movie and /genre/movie/list endpoints to gather movie metadata including overviews (plot summaries), poster image URLs, and their associated genre tags. The images were pulled from the provided poster paths and then were downloaded from TMDb's image server.
[TMDb's Movie Database](https://www.themoviedb.org/)
 
__Data Overview__
- The total number of samples we have is 14739.
- In our script to download the data we made sure that the distribution of our samples would be relatively even to avoid the problems that are associated with severe class imbalance. The distribution of these samples turned out like this:

<img src="/images/IMG_1341.png" alt="distribution" width="600" height="400">

<img src="/images/eda.png" alt="distribution" width="600" height="400">

- Some stadout features of our dataset is that we only downloaded movies from the year 2000 and later. This was purely a preferential choice. 
- Here is how our movies_dataset.csv is organized (id, title, overview, genres, poster_url) - and our posters were just downloaded to a directory called "movie_posters" with the naming convention "{id}".jpg. 

<img src="/images/IMG_9528.png" alt="distribution" width="600" height="400">

__Cleaning/Preparation__

A lot of the cleaning occured while downloading our data
{% highlight python %}
if not (movie_id and title and overview and genre_ids and poster_path):
    continue
{% endhighlight %}
- we ignore duplicated movies
{% highlight python %}
if movie_id in seen_ids:
    continue
{% endhighlight %}
- we ignore movies that have invalid genre ids
{% highlight python %}
valid_gids = [gid for gid in genre_ids if gid in genre_mapping]
if not valid_gids:
    continue
{% endhighlight %}
- we ensure each genre stays within a min/max sample cap
{% highlight python %}
under_min_gids = [gid for gid in prioritized_gids if genre_tag_counts[gid] < MIN_PER_GENRE]
selected_gids = under_min_gids if under_min_gids else [gid for gid in prioritized_gids if genre_tag_counts[gid] < MAX_PER_GENRE]
if not selected_gids:
    continue
{% endhighlight %}
- we ensure that only genres which meet the minimum sample count are included in the final dataset
{% highlight python %}
filtered_genres = [genre_mapping[gid] for gid in movie['genre_ids'] if gid in final_genres]
if not filtered_genres:
    continue
{% endhighlight %}

- because of our download methods, we actually had zero faulty entries.

__Text Cleaning and Tokenization__
- to further prepare the genres field
{% highlight python %}
df['genres'] = df['genres'].apply(lambda x: str(x) if isinstance(x, str) else "")
df['genres'] = df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')] if x else [])
{% endhighlight %}
- to clean and tokenize the overview (text summaries) we used this clean_text function and the nltk word_tokenize function
{% highlight python %}
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["overview"] = df["overview"].apply(clean_text)
tokenized = df["overview"].apply(word_tokenize)
{% endhighlight %}

__Vocabulary and Token Encoding__
- we made a vocabulary based on these tokenized words
{% highlight python %}
from collections import Counter
counter = Counter()
for tokens in tokenized:
    counter.update(tokens)

vocab = {word: i+2 for i, (word, _) in enumerate(counter.items())}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
{% endhighlight %}
- then we encoded the tokens into fixed length sequences
{% highlight python %}
def encode_tokens(tokens, vocab, max_len=200):
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(encoded) < max_len:
        encoded += [vocab["<PAD>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

df["input_ids"] = tokenized.apply(lambda x: encode_tokens(x, vocab))
{% endhighlight %}

__Split__
- our data was split as 80% training and 20% testing
{% highlight python %}
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
{% endhighlight %}
