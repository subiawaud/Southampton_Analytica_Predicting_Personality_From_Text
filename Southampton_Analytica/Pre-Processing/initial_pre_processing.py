#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import re
from nltk.corpus import stopwords
import nltk

import string
from emoji import UNICODE_EMOJI
from spellchecker import SpellChecker
from textblob import TextBlob
from textblob import Word
spell = SpellChecker()
nltk.download('brown')
from string import punctuation
from collections import Counter
#from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick, Tokenizer
from sklearn.feature_extraction.text import CountVectorizer


# # Data Prep
# 
# We start by loading the data, splitting the comments to be individual data points. 
# 

# In[2]:


mbti_data_original = pd.read_csv('mbti_data.csv')
yelp_data = pd.read_csv('yelp_comments_reduced.csv')


# In[3]:


mbti_split = mbti_data_original


# In[4]:


p_types = mbti_split['type'].unique()


# In[5]:


stop = stopwords.words('english')

def avg_word(sentence):
  words = sentence.split()
  if len(words) > 0:
      return round(sum(len(word) for word in words)/len(words),2)
  return 0

def get_misspelled_details(split_word, word_count):
    misspelled = spell.unknown(split_word)
    average_length_misspelled = round(sum(len(word) for word in misspelled)/len(misspelled),2)
    return(len(misspelled) / word_count), average_length_misspelled
    
def lemmatise_n_spell_check(word_list):
   # return [Word(spell.correction(word)).lemmatize() for word in word_list]
    return [Word(word).lemmatize() for word in word_list]

def remove_punctuation(entry):
    tab = str.maketrans(dict.fromkeys(string.punctuation))
    return entry.translate(tab)  

def count_emoji(word_list):
    emoji_count = 0
    for emoji in UNICODE_EMOJI:
        emoji_count += word_list.count(emoji)
    return emoji_count

def get_word_probabilities(split_word):
    word_probs = np.array([spell.word_probability(x) for x in split_word])
    word_probabilities = word_probs[word_probs > 0.00001]
    max_word_prob = max(word_probabilities)
    average_word_prob = np.mean(word_probabilities)
    lowest_word_prob = min(word_probabilities)
    std_word_prob = np.std(word_probabilities)
    return max_word_prob, average_word_prob, lowest_word_prob, std_word_prob
    
def get_sentiments(doc):
    
    text_blob = TextBlob(doc)
    polarity = []
    subjectivity = [] 
    for sentence in text_blob.sentences:
        polarity.append(sentence.sentiment.polarity)
        subjectivity.append(sentence.sentiment.subjectivity)

    
    return max(polarity), np.mean(polarity), min(polarity), max(subjectivity), np.mean(subjectivity), min(subjectivity)

# turn a doc into clean tokens
def clean_comment(doc):
   # for types in p_types:
   #     doc = re.sub(types,  'type',    doc)  
   #     doc = re.sub(types.lower(), 'type', doc)
    
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) ##counter

    doc = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', doc, flags=re.MULTILINE)

    max_polarity, average_polarity, min_polarity,  max_subjectivity, avg_subjectivity, min_subjectivity = get_sentiments(doc)
   
    punctuation_count = count(doc, string.punctuation)
    split_word = doc.split()
    word_count = len(split_word)
    char_count = len(doc)
    av_word = avg_word(doc)
    
    max_word_prob, average_word_prob, lowest_word_prob, std_word_prob = get_word_probabilities(split_word)

    
    emoji_count = count_emoji(doc)
    
    numerics = len([x for x in split_word if x.isdigit()])  
    stop_words = len([x for x in split_word if x in stop])
    upper = len([x for x in split_word if x.isupper() & len(x) > 1])
    

    doc = remove_punctuation(doc)
    tokens = re.sub("[^\w]", " ",  doc).split()
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop] 

    percentage_misspelled, avg_misspelled = get_misspelled_details(split_word, word_count)
    tokens = lemmatise_n_spell_check(tokens)

    return {'tokens' : tokens, 
            'max_polarity' : round(max_polarity,3),
            'average_polarity' : round(average_polarity,3),
            'min_polarity' : round(min_polarity,2),
            'max_subjectivity' : round(max_subjectivity,3),
            'average_subjectivity' : round(avg_subjectivity,3),
            'min_subjectivity' : round(min_subjectivity,3), 
            'misspelled%' : round(percentage_misspelled,3),
            'average_misspelled_word_length' : round(avg_misspelled,3),
            'emoji_%' : round(emoji_count / char_count, 3),
            'puncutation_%' : round(punctuation_count / char_count,3),
            'average word length' : av_word,
            'highest_word_probability' : round(max_word_prob,3),
            'average_word_probability' : round(average_word_prob,3),
            'lowest_word_probability' : round(lowest_word_prob,5),
            'std_word_probability' : round(std_word_prob,3),
            'number %' : round(numerics / word_count,3),
            'stop word %' : round(stop_words / word_count, 3),
            'upper word %' : round(upper / word_count ,3)}


# In[6]:


clean_comment(' I am so sad. This is truely terible and yet. I still have 1 or maybe even 2 emojis (bt I cant spell so well) 😁') 


# We next split the comments into word vectors, find the total number of unique words then use the md5 hash function to create an integer hash of each of the words 

# In[7]:


cleaned_yelp = yelp_data['text'].apply(lambda x : clean_comment(x))


# In[ ]:


cleaned_mbti = mbti_split['posts'].apply(lambda x : clean_comment(x))


# In[ ]:


def data_frame_the_cleaned_data(data, cleaned):    
    data['posts'] = [item['tokens'] for item in cleaned]
    data['posts'].replace('[]', np.nan, inplace=True)
    data['joined_comment'] = mbti_split['posts'].apply(lambda x: " ".join(x))
    data.dropna(subset=['posts'], inplace=True)

    data['max_polarity'] = [item['max_polarity'] for item in cleaned]
    data['average_polarity'] = [item['average_polarity'] for item in cleaned]
    data['min_polarity'] = [item['min_polarity'] for item in cleaned]

    data['max_subjectivity'] = [item['max_subjectivity'] for item in cleaned]
    data['average_subjectivity'] = [item['average_subjectivity'] for item in cleaned]
    data['min_subjectivity'] = [item['min_subjectivity'] for item in cleaned]

    data['noun_phase_%'] = [item['noun_phase_%'] for item in cleaned]
    data['misspelled%'] = [item['misspelled%'] for item in cleaned]
    data['average_misspelled_word_length'] = [item['average_misspelled_word_length'] for item in cleaned]
    data['emoji_%'] = [item['emoji_%'] for item in cleaned]
    data['puncutation_%'] = [item['puncutation_%'] for item in cleaned]

    data['average word length'] = [item['average word length'] for item in cleaned]
    data['highest_word_probability'] = [item['highest_word_probability'] for item in cleaned]
    data['average_word_probability'] = [item['average_word_probability'] for item in cleaned]
    data['std_word_probability'] = [item['std_word_probability'] for item in cleaned]

    data['number %'] = [item['number %'] for item in cleaned]
    data['stop word %'] = [item['stop word %'] for item in cleaned]
    data[ 'upper word %'] = [item[ 'upper word %'] for item in cleaned]
    return data


# In[ ]:


mbti_split = data_frame_the_cleaned_data(mbti_split, cleaned_mbti) 
yelp_data = data_frame_the_cleaned_data(yelp_data, cleaned_yelp) 


# In[ ]:



print(mbti_split.head())


# In[ ]:


b_o_w_vec = CountVectorizer(max_features=5000,  lowercase=True, ngram_range=(1,1), analyzer = "word")
b_o_w_mbti = b_o_w_vec.fit_transform(mbti_split['joined_comment'])
b_o_w_yelp = b_o_w_vec.transform(yelp_data['joined_comment'])


# In[ ]:


print(b_o_w_vec.vocabulary_)


# In[ ]:


b_o_w_processed_mbti = pd.DataFrame(b_o_w_mbti.toarray(), columns = b_o_w_vec.get_feature_names())
b_o_w_processed_yelp = pd.DataFrame(b_o_w_yelp.toarray(), columns = b_o_w_vec.get_feature_names())


# In[ ]:


def split_type(x):
    return [x[0], x[1], x[2], x[3]]


# In[ ]:


t = mbti_split['type'].apply(lambda x : split_type(x))
mbti_split['t1'] = [item[0] for item in t]
mbti_split['t2'] = [item[1] for item in t]
mbti_split['t3'] = [item[2] for item in t]
mbti_split['t4'] = [item[3] for item in t]

mbti_split.head(1000)


# In[ ]:


mbti_processed = pd.concat([mbti_split, b_o_w_processed_mbti ], axis=1, sort=False)
yelp_processed = pd.concat([yelp_data, b_o_w_processed_yelp ], axis=1, sort=False)

mbti_processed.to_csv('mbti_processed.csv')
yelp_processed.to_csv('yelp_processed.csv')
