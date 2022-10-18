# CMSC516-Advanced-Natural-Language-Processing
This is a repository includes the sentiment analysis project implementation of CMSC516_Advanced NLP.

## Authors
Lavanya Thollamadugu and Haolin Tang Group Project at Virginia Commonwealth University.

## Overview
In this project, we will build a Logistics Regression (LR) and a Long Short-Term Memory (LSTM) for sentiment analysis on movie reviews. First, we will download the IMDB movie reviews and conduct the data preprocessing. Second, a LR model and a LSTM will be trained and tested. In addition, the performance of these two models will be compared. Third, we will apply the Twitter API to collect some tweets mentioning a specific movie and then feed them to the LR and LSTM models. Last, we will investigate the sentiment analysis results on the tweets.        

## Installation & Usage
We recommend to run the code in Google Colab while we provide two options to run our codes. 
* **Run in Google Colab:**
    - Sentiment analysis using Logistics Regression.
   (Run [IMDB_Sentiment Analysis_LR.ipynb](https://colab.research.google.com/drive/1l6apFGjgIOJBiuueaW6r9PmwqaNFQDGs?usp=sharing))
   - Sentiment analysis using Long Short-Term Memory.\
   (Run [IMDB_Sentiment Analysis_LSTM.ipynb](https://colab.research.google.com/drive/1A8Agdp63gJH4KNGwfxVs9wgOcw9wsd0c?usp=sharing), **NOTE** Please change the Colab runtime type to GPU)
* **Run in Local System:**
    - Install Andconda (See [Installation Guide](https://docs.continuum.io/anaconda/install/))
    - Create a new conda environment named `cmsc516` with the following command.\
      `conda env create -f environment.yml`
    - Activate the environment: `conda activate cmsc516`  
    - Install jupyter notebook: `conda install jupyter`
    - Run jupyter from system: `jupyter notebook`. Now, you can run the notebooks in your local system.

## Data
The IMDB movie reveiws dataset (See [IMDB dataset](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/data)) consists of 50,000 movie reviews. 25,000 reviews are positive and 25,000 reviews are negative. This dataset is split into 80% training and 20% testing. The tweets mentioning a specific movie are collected using the Twitter API Framework (See [twitter_api_framework.ipynb](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/twitter_api_framework.ipynb)) for validation.

For data preprocessing, we remove HTLM tags, special characters and stopwords using NLTK library. We also convert the texts to lower case and conduct stemming.     

## Method
* **Logistics Regression:** `model = LogisticRegression(penalty='l2')`
* **Long Short-Term Memory:**\
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
* **Test Reulsts on IMDB Dataset:** 
   - Logistic regression: \
    ![image](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/lr_results.png)
   - Long Short-Term Memory: \
    ![image](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/lstm_result.png)\
    ![image](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/lstm_plot.png)\
     Next, We collect tweets related to five movies and apply the trained logistics model to make sentiment analysis since the logistics regression can provide higher accuracy than the LSTM model on IMDB test data.
* **Validation Results on Collected Tweets:**
    ![image](https://github.com/HaolinTang/CMSC516-Advanced-Natural-Language-Processing/blob/main/lr_tweets_results.png)

## Discussion
1. The LR model achieves higher accuracy than the LSTM model on IMDB test data. However, we expected the LSTM model can output higher accuracy. The reason could be that the training dataset is not large enough and the LSTM model is underfitting.    
2. The LR model outputs the highest positive rate to Spider-man No Way Home, which matches the IMDB rating. The LR model outputs the lowest positive rate to Red Notice, while F9 has the worst IMDB rating.
3. There are two reasons cause the discrepancy:
   - Many tweets collected by hashtag search are not related to movie reviews closely. For example, some tweets are advertisements or invitations.
   - The tweets collected are imbalanced which means that the extracted tweets have a different ratio between positive reviews and negative reviews.
4. Stemming does not have a significant impact on the accuracy of models in this project.

## Future Work
1. We will collect more tweets for a specific movie to avoid underfitting and improve the test accuracy of LSTM model. Once we conclude a LSTM model with higher accuracy than LR model, we will make sentiment analysis using LSTM. 
2. We will attempt to remove the tweets unrelated to movie reviews to mitigate the adverse effect of data imbalance.  
3. We will apply a pre-trained word embedding method for feature representation and investigate the performance. 



## License
[MIT](https://choosealicense.com/licenses/mit/)
