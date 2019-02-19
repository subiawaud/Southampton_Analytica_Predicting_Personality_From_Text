# Southampton_Analytica_Predicting_Personality_From_Text

Group project for Foundations of Data Science Module 

## Objective 

We present a predictive model pipeline for determining the personality of business users utilising  comments supporting business review feedback. We use forum comments with attached Myers Briggs personality types in building the models with high accuracy and use these models to determine the personality types of Yelp business reviews. We demonstrate the utility of such a pipeline in a web application which allows businesses to understand its reviewer base across the personality types.   

## Members 

Team lead : Chris Culley, Members : Claudia Subia, Jak Hall, Fairouz, Sam Banks 

## Datasets used
* MBTI Kaggle  - https://www.kaggle.com/datasnaek/mbti-type/home
* Yelp Kaggle - https://www.kaggle.com/yelp-dataset/yelp-dataset/home

## Method

Stage one - Chris: 

* Preprocess and feature engineering of the text data from both datasets. We explore sentiment analysis, word probabilities, spelling error length and percentage, language use as well as the standard lemmatisation and vectorisation. 

Stage two - Claudia & Chris: 

* Build predictive models that can predict each of the personality letters that constitute a personality type using the labelled MBTI data. We achieve an accuracy rate of 85-95% depending on the personality letter. 

Stage three - Fairoux, Sam, Chris, Claudia:

* Apply the strongest models to the Yelp dataset reviews so that we can profile each of the reviewers for businesses. We give some analysis in a write up report. 

Stage four - Jak: 

* Build a prototype application demonstrating the capability. Concluded and presented. 


