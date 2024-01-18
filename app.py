import pickle
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
load_model=pickle.load(open("C:/Users/sloka/OneDrive/Preparation/spam mail detection/train_model.sav",'rb'))
vectorizer = pickle.load(open("vectorizer.sav", 'rb'))

def predictive(input_text):
    input_text_as_array = np.asarray(input_text)
    fea_input_text = vectorizer.transform(input_text_as_array)
    ans = load_model.predict(fea_input_text)
    # feature_ext=TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    # input_text1=[input_text]
    # fea_input_text=feature_ext.transform(input_text1)
    # ans=load_model.predict(fea_input_text)
    if ans[0]==0:
        return "spam mail"
    else:
        return "Not a spam mail"
    

def main():
    st.title("Web app")
    text=st.text_input("Enter the mail message ")
    answer=''

    if st.button('Result'):
        answer=predictive(text)
    st.success(answer)

if __name__=='__main__':
    main()