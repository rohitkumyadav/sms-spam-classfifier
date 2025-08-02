import streamlit as st
import pickle
import nltk
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() # stemming

nltk.download('punkt', quiet=True)
nltk.download('stopwords',quiet = True)

tfidf  = pickle.load(open('vectorizer.pkl','rb'))
model  = pickle.load(open('model.pkl','rb'))

st.title("Sms Classfier")

input_sms = st.text_input("Enter the message: ")

# Preproess
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

transform_sms =  transform_text(input_sms)

# word to num/vectorize
vector_input = tfidf.transform([transform_sms])

# predict
result = model.predict(vector_input)[0]

# Result
if result == 1:
    st.header('Spam')

else:
    st.header('Not Spam')

