import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

credit_card_fraud = pd.read_csv("C:\\Users\\satya\\OneDrive\\Desktop\\codesoft\\credict_card_fraud_dection\\creditcard.csv")

# separating the data for analysis
legit = credit_card_fraud[credit_card_fraud.Class == 0]
fraud = credit_card_fraud[credit_card_fraud.Class == 1]
normal_treans = legit.sample(n = 497)
final_datasets = pd.concat([normal_treans,fraud],axis = 0)

x = final_datasets.drop(columns='Class', axis=1)
y = final_datasets['Class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)

model1 = LogisticRegression()
model1.fit(x_train,y_train)
traing_score = model1.score(x_train,y_train)
print("train accuracy :",traing_score)

test_size  = model1.score(x_test,y_test)
print("test accuracy :",test_size)

model2 =  DecisionTreeClassifier()
model2.fit(x_train,y_train)

traing_score = model2.score(x_train,y_train)
print("train accuracy :",traing_score)

model4 = RandomForestClassifier()
model4.fit(x_train,y_train)
test_size  = model4.score(x_test,y_test)
print("test accuracy :",test_size)


 ###Create a Streamlit app
st.title("Credit Card Fraud Detection")

# Input fields for user to enter transaction data
amount = st.number_input("Amount")
country = st.selectbox("Country", ["US", "UK", "Canada"])
merchant = st.selectbox("Merchant", ["Amazon", "Google", "Apple"])

# Button to predict fraud
if st.button("Predict"):
    # Create a dataframe with the input data
    input_data = pd.DataFrame({"amount": [amount], "country": [country], "merchant": [merchant]})
    input_data = input_data.rename(columns={'amount': 'Amount', 'country': 'Country', 'merchant': 'Merchant'})

    # Predict fraud probability
    proba = model4.predict_proba(input_data)[:, 1]

    # Display the result
    if proba > 0.5:
        st.error("Transaction is likely fraudulent!")
    else:
        st.success("Transaction is likely legitimate!")




























