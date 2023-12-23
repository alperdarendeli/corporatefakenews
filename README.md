```
*************************************************************************
***               THE GEOGRAPHY OF CORPORATE FAKE NEWS                ***
***                                                                   ***
***     Alper DARENDELI (Nanyang Technology University Singapore)     ***
***     SUN Aixin (Nanyang Technology University Singapore)           ***
***     TAY Wee Peng (Nanyang Technology University Singapore)        ***
***                                                                   ***
*************************************************************************
```

## Content
- [A. Corporate Fake News Dataset](#a-corporate-fake-news-dataset)
- [B. Training, Prediction and Tweet Datasets](#b-training-prediction-and-tweet-datasets)
- [C. Location Prediction Model](#c-location-prediction-model)
- [D. Firm Unique Identifiers](#d-firm-unique-identifiers)

## A. Corporate Fake News Dataset

The dataset ([corporate_fake_news.csv](datasets/corporate_fake_news.csv)) contains 685 fact-checked claims about U.S. firms (Jan 2012 - Jun 2021). 
| Variable | Definitions |
| :------- | :---------- |
| published_date | Publication date of fact-checking article |
| source | Fact-checking organization |
| source_link | URL link of fact-checking article |
| firm_name | Company name |
| news_topic | Topical classification of the fact-checked claim	|
| rumor_source_name | The source of rumor as mentioned in fact-checking article |
| verdict | Overall conclusion about the accuracy or truthfulness of the claim being fact-checked |

## B. Training, Prediction and Tweet Datasets

Training, prediction and tweets datasets may be downloaded from [releases](https://github.com/alperdarendeli/corporatefakenews/releases/tag/v1.0.0), please unzip before use.
- Training dataset (training_dataset.csv) comprises a total of 3,927,563 Tweet IDs that were posted on Twitter from 2014 to 2019. The dataset contains Tweet IDs, country codes and city IDs. City IDs may be mapped to their city names and country, refer to [utils\cities_mapper_new.csv](utils/cities_mapper_new.csv).
- Prediction dataset (prediction_dataset.csv) comprises a total of 189,158 unique User IDs of Twitter authors, whose locations are predicted in the study. 
- Tweet dataset (tweets_dataset.csv) contains comprises a total of 342,818 unique Tweet IDs, which post claims verified by a fact-checking organization. The variables in the dataset are Tweet ID, company name and the verdict of claim. 

## C. Location Prediction Model 

To replicate results, data from Twitter API is assumed to have been downloaded. Please ensure that the datasets have the following fields:
```
  Training dataset   :: tweet_id, country_code, city, tweet_text, tweet_lang, user_name, user_description, user_location
  Prediction dataset :: user_id, tweet_id, tweet_text, tweet_lang, user_name, user_description, user_location
```
**Part A: Training dataset**

1. Run `python a1_dataset_preprocessing.py` to clean the text data after modifying line 14 with the correct file path.  
2. Run `python a2_build_tokenizer.py -feat all` to train the BertTokenizerFast.
3. Run `python a3_train_optimized.py -gpu 0 -feat all -lr 3 -ndim 100 -nlstm 2` to train the model.
4. Run `python a4_lstm_test.py -gpu 0 -exp features_all_lr_1e-3_ndim_100_nlstm_2 -feat all -split val` to evaluate the validation set using the model's final weights.
5. Run `python a5_summarize_results.py` to print the metrics of the particular experiment. Modify line 39 to the correct experiment name.

**Part B: Prediction dataset**

1. Run `python b1_dataset_preprocessing.py` to clean the text data after modifying line 13 with the correct file path.  
2. Run `python b2_lstm_infer.py -SEED 0 -ndim 100 -nlstm 2` to prediction locations using the model.
3. Run `python b3_finalise_output.py` to map city predictions on tweets to final country predictions of users.  
   Results will be stored at `seeded-0/results/authlist/authlist_user_preds_ndim_2_nlstm_100.csv`.

## D. Firm Unique Identifiers

Accessing firm-specific data (i.e., COMPUSTAT, CRSP, Thomson Reuters 13F) requires a subscription to Wharton Research Data Services (WRDS). We provide a list of COMPUSTAT company unique identifiers ([gvkey.csv](datasets/gvkey.csv)) in the sample to facilitate replication of the empirical results. Supplementary Appendix 5 of the paper provides details about variable definitions and data sources. 

