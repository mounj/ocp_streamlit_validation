import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import pickle
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import plotly.graph_objs as go

class State:
    def __init__(self, path='state.pickle', default_state_class=dict):
        self.path = path
        self.default_state_class = default_state_class

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as inf:
                self.state = pickle.load(inf)
        else:
            self.state = self.default_state_class()

    def get_state(self):
        return self.state

    def save(self):
        with open(self.path, 'wb') as outf:
            pickle.dump(self.state, outf)

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

def app():
    store = State()
    store.load()

    name = store.get_state().get('name', None)
    if name:
        st.text(f'Hello {name}')
    else:
        ########################################################
        # Loading images to the website
        ########################################################
        image = Image.open("images/credit.jpg")
        
        st.title('Bienvenue sur Octroi de crédit !')
        current_path = os.getcwd()
        credit_path = os.path.join(current_path, 'classifier.pkl')
        with open(credit_path, 'rb') as handle:
            model = pickle.load(handle)
            
        #@st.cache()
        def prediction(X):
            prediction = model.predict(X)
            return prediction 
        
        def chargement_data(path):
            dataframe = pd.read_csv(path)
            liste_id = dataframe['SK_ID_CURR'].tolist()
            return dataframe, liste_id
        
        examples_file = 'df1.csv'
        dataframe, liste_id = chargement_data(examples_file)

        st.image(image)
        #st.sidebar.markdown("🛰️ **Navigation**")

        id_input = st.selectbox(
        'Choisissez le client que vous souhaitez visualiser',
        liste_id)

    
        client_infos = dataframe[dataframe['SK_ID_CURR'] == id_input].drop(
        ['SK_ID_CURR'], axis=1)
        #client_infos = client_infos.to_dict('record')[0]
        client_infos.to_dict(orient = 'records')
    
        result =""
        if st.button("Predict"):
        # correction 07/07 pour déceler les faux positifs
            X1 = dataframe[dataframe['SK_ID_CURR'] == id_input]    
            X = X1[['CODE_GENDER', 
                    'AGE',
                    'CNT_CHILDREN', 
                    'DEF_30_CNT_SOCIAL_CIRCLE',
                    'NAME_EDUCATION_TYPE_High education',  
                    'NAME_EDUCATION_TYPE_Low education',  
                    'NAME_EDUCATION_TYPE_Medium education',  
                    'ORGANIZATION_TYPE_Construction',  
                    'ORGANIZATION_TYPE_Electricity',  
                    'ORGANIZATION_TYPE_Government/Industry',  
                    'ORGANIZATION_TYPE_Medicine',  
                    'ORGANIZATION_TYPE_Other/Construction/Agriculture',  
                    'ORGANIZATION_TYPE_School',  
                    'ORGANIZATION_TYPE_Services',  
                    'ORGANIZATION_TYPE_Trade/Business', 
                    'OCCUPATION_TYPE_Accountants/HR staff/Managers', 
                    'OCCUPATION_TYPE_Core/Sales staff',  
                    'OCCUPATION_TYPE_Laborers',  
                    'OCCUPATION_TYPE_Medicine staff',  
                    'OCCUPATION_TYPE_Private service staff' , 
                    'OCCUPATION_TYPE_Tech Staff',
                    'NAME_FAMILY_STATUS',
                    'AMT_INCOME_TOTAL',
                    'INCOME_CREDIT_PERC',
                    'DAYS_EMPLOYED_PERC',
                    'EXT_SOURCE_1',
                    'EXT_SOURCE_2',    
                    'EXT_SOURCE_3']]
        
            result = prediction(X)
        
            if  result == 1:
              if int(X1['TARGET']) == 1: 
                pred = 'Rejected (True Negative)'
              else:
                pred = 'Approved (False Negative)'
            else:
              if int(X1['TARGET']) == 1:
                pred = 'Rejected (False Positive)'
              else:
                pred = 'Approved (True Positive)'              
                   
            st.success('Your loan is {}'.format(pred))
        
        
        
        st.text(f'Please enter your name')
        name_input = st.text_input('your name')
        name = name_input

        if name != '':
            store.get_state()['name'] = name

        store.save()
        next_page = st.button('Next page')
        if next_page:
            rerun()

if __name__ == "__main__":
    app()