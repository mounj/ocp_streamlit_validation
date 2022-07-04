# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import pickle
import os
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objs as go


st.title('Bienvenue sur Octroi de crédit !')

# loading the trained model
#with open(r'C:\Users\Catherine\Credit\classifier.pkl', 'rb') 

current_path = os.getcwd()
credit_path = os.path.join(current_path, 'classifier.pkl')
with open(credit_path, 'rb') as handle:
    model = pickle.load(handle)

########################################################
# Loading images to the website
########################################################
image = Image.open("images/credit.jpg")

@st.cache()
def prediction(X):
    prediction = model.predict(X)
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred 

######################################
# Feature Selection Code
######################################
def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)
    
    

def main():
    def chargement_data(path):
        dataframe = pd.read_csv(path)
        liste_id = dataframe['SK_ID_CURR'].tolist()
        return dataframe, liste_id
        
    
    st.subheader("Prédictions de scoring client et positionnement dans l'ensemble des clients")

    examples_file = 'df1.csv'
    dataframe, liste_id = chargement_data(examples_file)

    st.sidebar.image(image)
    st.sidebar.markdown("🛰️ **Navigation**")

    id_input = st.sidebar.selectbox(
        'Choisissez le client que vous souhaitez visualiser',
        liste_id)

    focus_var = st.sidebar.selectbox('Choisissez la variable de focus',
                                     ['EXT_SOURCE_1',
                                      'EXT_SOURCE_2',
                                      'EXT_SOURCE_3'])

    client_infos = dataframe[dataframe['SK_ID_CURR'] == id_input].drop(
        ['SK_ID_CURR'], axis=1)
    client_infos = client_infos.to_dict('record')[0]
    
    result =""
    
    if st.sidebar.button("Predict"):
        X = dataframe[dataframe['SK_ID_CURR'] == id_input]    
        X = X[['CODE_GENDER', 
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
        if result == 0:
         result = 'Rejected'
        else:
         result = 'Approved'
         
        st.success('Your loan is {}'.format(result))

        # informations du client
        st.header("Informations du client")
        examples_file = 'application_API.csv'
        application, liste_id = chargement_data(examples_file)
        application = application[~((application['EXT_SOURCE_1'].isnull()))]
        X1 = application[application['SK_ID_CURR'] == id_input]  
        st.write(X1)
        
        st.header("Graphique d'explication")
        feat_importances = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.subheader('Random Forest Classifier:')
        impPlot(feat_importances, 'Random Forest Classifier')
        #trace = go.Bar(x=feat_importances.values,y=feat_importances.keys(),showlegend = True)
        #layout = go.Layout(title = "Importance des features")
        #data = [trace]
        #fig = go.Figure(data=data,layout=layout)
        #st.plotly_chart(fig)
                   
        
        st.header("Positionnement du client")

        if focus_var == 'EXT_SOURCE_1':           
           source1 = application[['TARGET', 'EXT_SOURCE_1']]
           source1['SOURCE_BINNED'] = pd.cut(source1['EXT_SOURCE_1'], bins = np.linspace(0.2, 0.8, num = 15))
           ext_source1  = source1.groupby('SOURCE_BINNED').mean()               
           trace = go.Bar(x=ext_source1.index.astype(str),y=ext_source1['TARGET'].values*100,showlegend = True)
           layout = go.Layout(title = "Difficulté de payer en fonction des tranches de source1")
           data = [trace]
           fig = go.Figure(data=data,layout=layout)
           st.plotly_chart(fig)
            
        if focus_var == 'EXT_SOURCE_2':           
           #fig2 = px.density_heatmap(data_frame= dataframe, y="TARGET", x="EXT_SOURCE_2")
           #st.write(fig2) 
           source2 = application[['TARGET', 'EXT_SOURCE_2']]
           source2['SOURCE_BINNED'] = pd.cut(source2['EXT_SOURCE_2'], bins = np.linspace(0.2, 0.8, num = 15))
           ext_source2  = source2.groupby('SOURCE_BINNED').mean()                      
           trace = go.Bar(x=ext_source2.index.astype(str),y=ext_source2['TARGET'].values*100,showlegend = True)
           layout = go.Layout(title = "Difficulté de payer en fonction des tranches de source2")
           data = [trace]
           fig = go.Figure(data=data,layout=layout)
           st.plotly_chart(fig) 
        
        if focus_var == 'EXT_SOURCE_3':
           source3 = application[['TARGET', 'EXT_SOURCE_3']]
           source3['SOURCE_BINNED'] = pd.cut(source3['EXT_SOURCE_3'], bins = np.linspace(0.2, 0.8, num = 15))
           ext_source3  = source3.groupby('SOURCE_BINNED').mean()                      
           trace = go.Bar(x=ext_source3.index.astype(str),y=ext_source3['TARGET'].values*100,showlegend = True)
           layout = go.Layout(title = "Difficulté de payer en fonction des tranches de source3")
           data = [trace]
           fig = go.Figure(data=data,layout=layout)
           st.plotly_chart(fig)
           

if __name__ == '__main__':
    main()
