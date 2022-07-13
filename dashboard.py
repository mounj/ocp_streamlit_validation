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
from sklearn.preprocessing import StandardScaler
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

scaler = StandardScaler()

########################################################
# Loading images to the website
########################################################
image = Image.open("images/credit.jpg")

@st.cache()
def prediction(X):
    prediction = model.predict(X)
    #if prediction == 0:
    #    pred = 'Approved'
    #else:
    #    pred = 'Rejected'
    return prediction 

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

    #focus_var = st.sidebar.selectbox('Choisissez la variable de focus',
    #                                 ['EXT_SOURCE_1',
    #                                  'EXT_SOURCE_2',
    #                                  'EXT_SOURCE_3'])

    client_infos = dataframe[dataframe['SK_ID_CURR'] == id_input].drop(
        ['SK_ID_CURR'], axis=1)
    #client_infos = client_infos.to_dict('record')[0]
    client_infos.to_dict(orient = 'records')
    
    result =""
    
    if st.sidebar.button("Predict"):
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

        # informations du client
        st.header("Informations du client")
        examples_file = 'application_API.csv'
        application, liste_id = chargement_data(examples_file)
        application = application[~((application['EXT_SOURCE_1'].isnull()))]
        X1 = application[application['SK_ID_CURR'] == id_input]  
        st.write(X1)
        
        # SHAP variables locales 
        st.header("Graphique d'explication")
        feat_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.subheader('Random Forest Classifier:')
        impPlot(feat_importances, 'Random Forest Classifier')
        #trace = go.Bar(x=feat_importances.values,y=feat_importances.keys(),showlegend = True)
        #layout = go.Layout(title = "Importance des features")
        #data = [trace]
        #fig = go.Figure(data=data,layout=layout)
        #st.plotly_chart(fig)
                   
        
        st.header("Transparence des données client")

        #if focus_var == 'EXT_SOURCE_1':           
        #   source1 = application[['TARGET', 'EXT_SOURCE_1']]
        #   source1['SOURCE_BINNED'] = pd.cut(source1['EXT_SOURCE_1'], bins = np.linspace(0.2, 0.8, num = 15))
        #   ext_source1  = source1.groupby('SOURCE_BINNED').mean()               
        #   trace = go.Bar(x=ext_source1.index.astype(str),y=ext_source1['TARGET'].values*100,showlegend = True)
        #   layout = go.Layout(title = "% Faillite par tranches de source1")
        #   data = [trace]
        #   fig = go.Figure(data=data,layout=layout)
        #   st.plotly_chart(fig)
        
        # Saisie des informations Client     
        CODE_GENDER = st.selectbox("CODE_GENDER",options=['M' , 'F'])
        AGE = st.slider("AGE", 1, 100,1)
        CNT_CHILDREN = st.slider("CNT_CHILDREN", 1, 5,1)
        NAME_EDUCATION_TYPE = st.selectbox("NAME_EDUCATION_TYPE",options=['Low education','Medium education','High education'])
        ORGANIZATION_TYPE = st.selectbox("ORGANIZATION_TYPE",options=['Construction', 'Electricity', 'Government/Industry', 'Medicine', 
                                                                      'Other/Construction/Agriculture', 'School', 'Services', 
                                                                      'Trade/Business'])
        OCCUPATION_TYPE = st.selectbox("OCCUPATION_TYPE",options=['Accountants/HR staff/Managers','Core/Sales staff','Laborers',
                                                                  'Medicine staff','Private service staff' , 'Tech Staff'])
        NAME_FAMILY_STATUS = st.selectbox("NAME_FAMILY_STATUS",options=['Single' , 'Married'])
        AMT_INCOME_TOTAL = st.slider("AMT_INCOME_TOTAL", 1, 500000,1000)
        INCOME_CREDIT_PERC = st.slider("INCOME_CREDIT_PERC", 1, 100,10)
        DAYS_EMPLOYED_PERC = st.slider("DAYS_EMPLOYED_PERC", 1, 100,10)
        EXT_SOURCE_1 = st.slider("EXT_SOURCE_1", 1, 100,10)
        EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", 1, 100,10)
        EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", 1, 100,10)  
        
        # Scaling pour prédiction
        CODE_GENDER = 0 if  CODE_GENDER == 'M' else 1
        
        NAME_EDUCATION_TYPE_Low_education , NAME_EDUCATION_TYPE_Medium_education , NAME_EDUCATION_TYPE_High_education = 0,0,0
        if NAME_EDUCATION_TYPE == 'Low education':
            NAME_EDUCATION_TYPE_Low_education = 1
        elif NAME_EDUCATION_TYPE == 'Medium education':
            NAME_EDUCATION_TYPE_Medium_education = 1
        else:
            NAME_EDUCATION_TYPE_High_education = 1
            
        ORGANIZATION_TYPE_Construction, ORGANIZATION_TYPE_Electricity, ORGANIZATION_TYPE_Government_Industry = 0,0,0
        ORGANIZATION_TYPE_Medicine, ORGANIZATION_TYPE_Other_Construction_Agriculture, ORGANIZATION_TYPE_School = 0,0,0
        ORGANIZATION_TYPE_Services, ORGANIZATION_TYPE_Trade_Business = 0,0 
        if ORGANIZATION_TYPE == 'Construction':
            ORGANIZATION_TYPE_Construction = 1
        elif ORGANIZATION_TYPE == 'Electricity':
            ORGANIZATION_TYPE_Electricity = 1
        elif ORGANIZATION_TYPE ==  'Government/Industry':   
            ORGANIZATION_TYPE_Government_Industry = 1 
        elif ORGANIZATION_TYPE == 'Medicine':    
            ORGANIZATION_TYPE_Medicine = 1
        elif ORGANIZATION_TYPE == 'Other/Construction/Agriculture':
            ORGANIZATION_TYPE_Other_Construction_Agriculture = 1
        elif ORGANIZATION_TYPE ==  'School': 
            ORGANIZATION_TYPE_School = 1
        elif ORGANIZATION_TYPE == 'Services':
            ORGANIZATION_TYPE_Services = 1
        elif ORGANIZATION_TYPE == 'Trade/Business':
            ORGANIZATION_TYPE_Trade_Business = 1
            
        OCCUPATION_TYPE_Accountants_HR_staff_Managers, OCCUPATION_TYPE_Core_Sales_staff, OCCUPATION_TYPE_Laborers = 0,0,0  
        OCCUPATION_TYPE_Medicine_staff, OCCUPATION_TYPE_Private_service_staff, OCCUPATION_TYPE_Tech_Staff = 0,0,0
        if OCCUPATION_TYPE == 'Accountants/HR staff/Managers':
           OCCUPATION_TYPE_Accountants_HR_staff_Managers = 1
        elif OCCUPATION_TYPE == 'Core/Sales staff':
           OCCUPATION_TYPE_Core_Sales_staff = 1
        elif OCCUPATION_TYPE == 'Laborers':
           OCCUPATION_TYPE_Laborers = 1
        elif OCCUPATION_TYPE == 'Medicine staff':
           OCCUPATION_TYPE_Medicine_staff = 1
        elif OCCUPATION_TYPE == 'Private service staff':
           OCCUPATION_TYPE_Private_service_staff = 1 
        elif OCCUPATION_TYPE ==  'Tech Staff':
           OCCUPATION_TYPE_Tech_Staff = 1
        
        NAME_FAMILY_STATUS = 0 if  NAME_FAMILY_STATUS == 'Single' else 1
        
        
        input_data = scaler.transform([[CODE_GENDER,
                                        AGE, 
                                        CNT_CHILDREN,
                                        NAME_EDUCATION_TYPE_Low_education, 
                                        NAME_EDUCATION_TYPE_Medium_education, 
                                        NAME_EDUCATION_TYPE_High_education,
                                        ORGANIZATION_TYPE_Construction, 
                                        ORGANIZATION_TYPE_Electricity, 
                                        ORGANIZATION_TYPE_Government_Industry,
                                        ORGANIZATION_TYPE_Medicine, 
                                        ORGANIZATION_TYPE_Other_Construction_Agriculture, 
                                        ORGANIZATION_TYPE_School,
                                        ORGANIZATION_TYPE_Services, 
                                        ORGANIZATION_TYPE_Trade_Business,
                                        OCCUPATION_TYPE_Accountants_HR_staff_Managers,
                                        OCCUPATION_TYPE_Core_Sales_staff, 
                                        OCCUPATION_TYPE_Laborers,
                                        OCCUPATION_TYPE_Medicine_staff, 
                                        OCCUPATION_TYPE_Private_service_staff, 
                                        OCCUPATION_TYPE_Tech_Staff,
                                        NAME_FAMILY_STATUS,
                                        AMT_INCOME_TOTAL,
                                        INCOME_CREDIT_PERC,
                                        DAYS_EMPLOYED_PERC,
                                        EXT_SOURCE_1,
                                        EXT_SOURCE_2,    
                                        EXT_SOURCE_3
                                        ]])
        
        transparence = prediction(input_data)
        
        predict_probability = model.predict_proba(input_data)
        
        if transparence[0] == 1:
           st.subheader('Client {} aurait une probabilité de faillite de {}%'.format(name , round(predict_probability[0][1]*100 , 3)))
        else:
           st.subheader('Client {} aurait une probabilité de remboursement de {}%'.format(name, round(predict_probability[0][0]*100 
                                                                                                            , 3)))            
        
        
           

if __name__ == '__main__':
    main()
