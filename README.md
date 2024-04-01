```
               *************************************************************************
               ***               THE GEOGRAPHY OF CORPORATE FAKE NEWS                ***
               ***                                                                   ***
               ***     Alper DARENDELI (Nanyang Technology University Singapore)     ***
               ***     SUN Aixin (Nanyang Technology University Singapore)           ***
               ***     TAY Wee Peng (Nanyang Technology University Singapore)        ***
               ***                                                                   ***
               *************************************************************************

                                      PLOS ONE, Forthcoming

```

## Content
- [A. Corporate Fake News Dataset](#a-corporate-fake-news-dataset)
- [B. Training, Prediction and Tweet Datasets](#b-training-prediction-and-tweet-datasets)
- [C. COMPUSTAT Firm Identifiers](#d-compustat-Firm-identifiers)

## A. Corporate Fake News Dataset

The dataset ([corporate_fake_news.csv](datasets/corporate_fake_news.csv)) contains 685 fact-checked claims about U.S. firms (Jan 2012 - Jun 2021). 
| Variable | Definitions |
| :------- | :---------- |
| published_date | Publication date of fact-checking article |
| source | Name of fact-checking organization |
| source_link | URL link of fact-checking article |
| firm_name | Company name |
| news_topic | Topical classification of fact-checking article	|
| rumor_source_name | The source of rumor as mentioned in fact-checking article |
| verdict | Overall conclusion about the accuracy or truthfulness of the claim being fact-checked |

## B. Training, Prediction and Tweet Datasets

Training, prediction and tweets datasets may be downloaded from [here](https://github.com/alperdarendeli/corporatefakenews/releases/tag/v1.0.0). Please unzip before use.
- Training dataset (training_dataset.csv) comprises a total of 3,927,563 Tweet IDs that were posted on Twitter from 2014 to 2019. The dataset contains Tweet IDs, Country Codes and City IDs. To map City IDs  to their respective city names and countries, please refer to [utils\cities_mapper_new.csv](utils/cities_mapper_new.csv).
- Prediction dataset (prediction_dataset.csv) comprises a total of 189,158 unique User IDs, whose locations are predicted in the study. 
- Tweet dataset (tweets_dataset.csv) comprises a total of 342,818 unique Tweet IDs, which post claims verified by a fact-checking organization. The variables in the dataset are Tweet ID, Company Name and Verdict of Claim. 

## C. COMPUSTAT Firm Identifiers

Accessing firm-specific data (i.e., COMPUSTAT, CRSP, Thomson Reuters 13F) requires a subscription to Wharton Research Data Services (WRDS). We provide a list of COMPUSTAT company identifiers in the sample ([gvkey.csv](datasets/gvkey.csv)) to facilitate replication of the empirical results. Supplementary Appendix 5 of the paper provides details about variable definitions and data sources. 

