# CMSC516-Advanced-Natural-Language-Processing
This is a repository includes the sentiment analysis project implementation of CMSC516_Advanced NLP.

## Authors
Lavanya Thollamadugu and Haolin Tang Group Project at Virginia Commonwealth University.

## Overview
In this project, we will build a Logistics Regression (LR) and a Long Short-Term Memory (LSTM) for sentiment analysis on movie reviews. First, we will download the IMDB movie reviews and conduct the data preprocessing. Second, a LR model and a LSTM will be trained and tested. In addition, the performance of these two models will be compared. Third, we will apply the Twitter API to collect some tweets mentioning a specific movie and then feed them to the LR and LSTM models. Last, we will investigate the sentiment analysis results on the tweets.        

## Installation 



## Data
The [IMDB dataset](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/data) consists of 50,000 IMDB movie reviews. This dataset is split into 80% training and 20% testing. The tweets mentioning a specific movie are collected using the Twitter API Framework (See [twitter_api_framework.ipynb](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/twitter_api_framework.ipynb)) for validation.

For data preprocessing, we remove HTLM tags, special characters and stopwords using NLTK library. We also convert the texts to lower case and conduct stemming.     

## Method
* **Logistics Regression:** `model = LogisticRegression(penalty='l2')`
* **Long Short-Term Memory:**
 `SentimentRNN(`\
  `(embedding): Embedding(1001, 64)`\
  `(lstm): LSTM(64, 256, num_layers=2, batch_first=True)`\
  `(dropout): Dropout(p=0.3, inplace=False)`\
  `(fc): Linear(in_features=256, out_features=1, bias=True)`\
  `(sig): Sigmoid()`\
  `)`
  
  Training Details: 
  - `batch_size = 50`  
  - `epochs = 5`
  - `criterion = nn.BCELoss()`
  - `optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)`


## Results
* **Logistics Regression:** 
   - Test reulsts on IMDB dataset: \
    ![image](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/lr_results.png)\
   - Validation result on collected tweets:\


* **Long Short-Term Memory:**
## Discussion

## Future Work


## License
[MIT](https://choosealicense.com/licenses/mit/)
