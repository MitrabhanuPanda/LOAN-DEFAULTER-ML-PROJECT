import streamlit as st
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open(r"MODEL/model.pkl","rb"))
df=pickle.load(open(r"MODEL/df.pkl","rb"))
# df1=pd.read_csv(r"C:\Users\mitra\OneDrive\Desktop\CREDIT RISK\DATASET\1. LOAN DEFAULT DATA\final_data.csv")


st.title("LOAN DEFAULTER MODEL PREDICTION")

annual_income=st.number_input("PERSON ANNUAL INCOME",value=None,step=1)
credit_score= st.number_input("ENTER THE CREDIT SCORE OF THE PERSON",value=None,step=1, max_value=900,min_value=300)
previous_defult=st.number_input("HOW MANY TIME PERSON IS PREVIOUSLY DEFAULT",value=None,step=1, max_value=15,min_value=0)
credit_utilization = st.number_input("CREDIT UTILIZATION OF PERSON",value=None, max_value=1.00,min_value=0.00)
inquiry_counts=st.number_input("HOW MANY TIME PERSON IS INQUIRED",value=None,step=1, max_value=15,min_value=0)
debt_to_income_ratio = st.number_input("DTI OF PERSON",value=None, max_value=1.00,min_value=0.00)



if st.button("PREDICTION"):
    new_query=np.array([[annual_income,credit_score,previous_defult,credit_utilization,inquiry_counts,debt_to_income_ratio]])
    new_query=new_query.reshape(1,6)

    prob_pred=model.predict_proba(new_query)
    
    pred_out=np.where(prob_pred[:, 1]>=0.90,1,0)

    if int(pred_out)==1:
        st.title(f"THE PREDICTION IS : DEFAULT" )
    else:
        st.title(f"THE PREDICTION IS : NON DEFAULT" )
