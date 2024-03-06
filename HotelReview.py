
import sklearn
from typing import Counter
import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
stop_words_Keywords = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
import string

#Load Model
model=pickle.load(open("Hotel_Review_Analysis.pkl","rb"))
#Attri=pickle.load(open("cleaned_data.pkl","rb" ))
#creating a Function for data cleaning
lemmatizer=WordNetLemmatizer()
def clean_text(text):
  text=re.sub(r'\w*\d\w*','',str(text)).strip()    # removing numbers attached to the words
  text = re.sub("[\d]+", "", str(text))            # removing strings containg unwanted digits
  text = text.translate(str.maketrans('','',string.punctuation))   # removing punchuations
  text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))  # removing puncuations
  text = re.sub(r'[^\w\s]', " ", str(text))         # removing white spaces
  text = ' '.join( [w for w in text.split() if len(w)>1] )   #removing single characters
  text=text.split()
  text=" ".join([word for word in text if word not in stop_words_Keywords])   # removing stopwords
  text=nltk.word_tokenize(text)  # Tokenizing
  text=" ".join([lemmatizer.lemmatize(w,"v") for w in text])    # applying lemmatization
  text=" ".join(dict.fromkeys(text.split()))  # remove duplicate words
  return text

# Creating title
st.title (":blue[Hotel review Sentiment Analysis]")


review=st.text_input("Enter your review")

submit=st.button("SUBMIT")

if submit:
  prediction=model.predict([review])
  print(prediction)
  if prediction[0]==1:
    st.success("Positive Review")
  else:
      st.warning("Negative Review")

  cleaned_review=[]  # list of cleaned reviews
  for i in [review]:
    cleaned_review.append(clean_text(i))

  CV=CountVectorizer()
  P=CV.fit_transform(cleaned_review)
  features=CV.get_feature_names_out()
  st.text(features)

