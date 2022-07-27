# -*- coding: utf-8 -*-


# Core packages
import streamlit as st
from streamlit_option_menu import option_menu

# EDA packages
import pandas as pd
import numpy as np

# Utils
import os
import io
# import webbrowser
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from PIL import Image

home_img = Image.open("assets/1.jpg")
home_img2 = Image.open("assets/2.jpg")


# Data viz
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
st.set_option('deprecation.showPyplotGlobalUse', False)

# ML Interpretation
import lime
import lime.lime_tabular

feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue','spiders', 'ascites',
                   'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime',
                   'histology']

# Setting value as per our data since our dataset has values in 1 and 2
gender_dict = {'male':1, 'female':2} 
feature_dict = {'No':1, 'Yes':2} 

# For getting keys, feature values
def get_value(val, my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value 

def get_key(val, my_dict):
    for key,value in my_dict.items():
        if val == key:
            return key
    
def get_feature_value(val):
    feature_dict = {'No':1, 'Yes':2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model
        

img = Image.open("assets/icon.jpg") 

st.set_page_config(page_title="Disease Prediction Web App", 
                   page_icon= img,
                   layout='centered')


def main():
    
    df = pd.read_csv("Datasets/data.csv")
    df1 = pd.read_csv("Datasets/balanced_data.csv")
    df1.drop('Unnamed: 0', axis=1, inplace=True)
    menu = ['Home', 'Quick Explore', 'Plot', 'Prediction']
    
    with st.sidebar:
        
        option = option_menu(menu_title= 'Menu' ,
                             options=['Home', 'Quick Explore', 'Plot', 'Prediction'],
                             icons=['house', 'circle', 'bar-chart-line', 'caret-right'],
                             styles={
                                     'Container':{'padding':'0!important', 'background-color':'#fafafa'},
                                     'icon':{'color':'black', 'font-size':'25px'},
                                     'nav-link':{
                                                 'font-size':'20px',
                                                 'text-align':'left',
                                                 'margin':'0px',
                                                 '--hover-color': '#eee',
                                                 },
                                     'nav-link-selected':{'background-color':'peach'},
                                     },                            
                            )        

    if option == 'Home':    
        
        st.title("Disease Prediction Web App")
        st.image(home_img)
        st.subheader('Hepatitis B')

        if st.checkbox('Overview'):
            
    
            st.markdown('#### Overview -')
            st.write(
                'Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease.'
                ' This disease is most commonly spread by exposure to infected bodily fluids, by blood products (unclean needles or unscreened blood), By having unprotected vaginal, anal or oral sex, By mother to baby by pregnancy, labour or nursing..'
                ' The condition often clears up on its own. Chronic cases require medication and possibly a liver transplant.'
                )
            st.text('More than 1 million cases per year (India)')
            st.write('###### Spreads by sexual contact')
            st.write('###### Preventable by vaccine')
            st.write('###### Treatable by a medical professional')
            st.write('###### Requires a medical diagnosis')
            st.write('###### Lab tests or imaging always required')
            #st.image(home_img2)
            
        if st.checkbox('Symptoms'):
            st.markdown('#### Symptoms -')
            st.markdown('###### Requires a medical diagnosis ')
            st.write("Symptoms are variable and include yellowing of the eyes, abdominal pain and dark urine. Some people, particularly children, don't experience any symptoms. In chronic cases, liver failure, cancer or scarring can occur.")
            st.markdown('###### Can have no symptoms, but people may experience:')
            st.write('Whole body: fatigue or malaise')
            st.write('Skin: web of swollen blood vessels in the skin or yellow skin and eyes')
            
        if st.checkbox('Treatment'):
            st.markdown('#### Treatment -')
            st.markdown('###### Treatment depends on severity')
            st.write('The condition often clears up on its own. Chronic cases require medication and possibly a liver transplant.')
            st.text('Medications --> Antiviral Drugs')
            st.text('Self-care --> Avoid Alcohol')    
                
        
    elif option == 'Quick Explore':
        progress = st.progress(0)
        for i in range(100):
                time.sleep(0.02)
                progress.progress(i+1)
                
        st.subheader('Quick  Explore')
        
        if st.checkbox('Raw Data'):
            st.subheader('Raw Data')
            st.info('This data is Uncleaned and Imbalanced.')
            st.write(df)
            st.write('Shape of the raw data-',df.shape)
        
        if st.checkbox('Clean Data'):
            st.subheader('Clean Dataset Quick Look:')
            st.info('This data is Cleaned, Balanced and only has Important Features.')
            st.write(df1)
            st.write('Shape of the clean data-',df1.shape)
        
        if st.button("Show Columns"):
            st.subheader('Columns List')
            all_columns = df1.columns.to_list()
            st.write(all_columns)
        
        if st.button('Basic Information'):
            st.subheader('Basic Information of Data')
            #st.write(df1.info())
            buffer = io.StringIO()
            df1.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
        if st.button('Statistical Description'):
            st.subheader('Statistical Data Descripition')
            st.write(df1.describe())
        
        if st.button('Missing Values?'):
            st.subheader('Missing values')
            miss_val = df1.isnull().sum()
            st.dataframe(miss_val, width=800, height=445)

    elif option == 'Plot':
        progress = st.progress(0)
        for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
        
        st.subheader("Data Visualization")
        df1 = pd.read_csv("D:/Softwares/DS/Project/Hepatitis/Original data/balanced_data.csv")
        df1_numeric = df1[['age','bilirubin','alk_phosphate','sgot','albumin','protime']]
        df1_categorical = df1[['sex','steroid','antivirals','fatigue','spiders','ascites','varices','histology','class']]
                            
        if st.checkbox('Count Plot'):
            st.subheader('Count Plot')
            st.info("If error, please adjust column name on below panel. [Male-1 , Female-2 ; No-1 , Yes-2]")
            column_count_plot = st.selectbox("Choose a column to plot count. Try Selecting Sex ",df1_categorical.columns)
            hue_opt = st.selectbox("Optional variables. Try Selecting Class ",df1_categorical.columns.insert(0,None))
            fig = sns.countplot(x=column_count_plot,data=df1,hue=hue_opt,palette='Set1')
            st.pyplot()
        
        if st.checkbox('Histogram'):
            st.subheader('Histogram | Distplot')
            st.info("If error, please adjust column name on below panel.")
            column_dist_plot = st.selectbox("Select a feature",df1_numeric.columns)
            fig = px.histogram(df1[column_dist_plot])
            st.plotly_chart(fig)

        if st.checkbox('Line Chart'):
            st.subheader('Line Chart')
            all_columns = df1.columns.to_list()
            feature_choice = st.multiselect('Select a feature', all_columns)
            new_df1 = df1[feature_choice]
            st.line_chart(new_df1) 

            
    elif option == 'Prediction':
        with st.spinner('Wait for it...'):
            time.sleep(3)
        
        st.subheader('Predictive Analytics')
        
        age = st.number_input('Age', 7, 80) # Since in our data the lower and upper limit for age is 7 and 80 resp.
        sex = st.selectbox('Sex', tuple(gender_dict.keys()))
        steroid = st.radio('Do you take Steroids?', tuple(feature_dict.keys()))
        antivirals = st.radio('Do you take Antivirals?', tuple(feature_dict.keys()))
        fatigue = st.radio('Do you have Fatigue?', tuple(feature_dict.keys()))
        spiders = st.radio('Presence of Spider Naive?', tuple(feature_dict.keys()))
        ascites = st.radio('Ascites', tuple(feature_dict.keys()))
        varices = st.radio('Presence of Varices', tuple(feature_dict.keys())) 
        bilirubin = st.number_input('Bilirubin Content',0.0,8.0)
        alk_phosphate = st.number_input('Alkaline Phosphate Content',0.0,296.0)
        sgot = st.number_input('Sgot',0.0,648.0)
        albumin = st.number_input('Albumin',0.0,6.4)
        protime = st.number_input('Prothrombine Time',0.0,100.0)
        histology = st.radio('Histology', tuple(feature_dict.keys()))
        
        # Each above entries will get stored in the list which will be send to our ML model
        feature_list = [age,get_value(sex,gender_dict),get_feature_value(steroid),get_feature_value(antivirals),get_feature_value(fatigue),get_feature_value(spiders),get_feature_value(ascites),get_feature_value(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_feature_value(histology)]
        #st.write(feature_list) # Here we can see that our entries which were in str(like sex) got converted into numeric aand now will be send to ML model
        single_sample = np.array(feature_list).reshape(1,-1) #reshaping so that our model doest not throw any error
        
        # This values will be seen by a person like ok I have selected this values
        pretty_result = {'age':age, 'sex':sex, 'steroid':steroid, 'antivirals':antivirals, 'spiders':spiders, 'ascites':ascites, 'varices':varices, 'bilirubin':bilirubin, 'alk_phosphate':alk_phosphate, 'sgot':sgot, 'albumin':albumin, 'protime':protime, 'histology':histology}
        st.json(pretty_result)
        
        # ML models
        model_choice = st.selectbox('Select Model',['Random Forest'])
        if st.button('Predict'):
            if model_choice == 'Random Forest':
                loaded_model = load_model("Random_Forest_Model.pkl")
                prediction =loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)
           
            if prediction == 1:
                st.warning('Patient Dies')
                st.subheader('Prescriptive Analysis')
                st.markdown('##### Recommended life style modification -')
                st.markdown('###### Get rest, Exercise Daily')
                
            else:
                st.success('Patient Lives')
                
            pred_probability_score = {'Die':pred_prob[0][0]*100, 'Live':pred_prob[0][1]*100}
            st.subheader('Prediction Probability Score:')
            st.json(pred_probability_score)
   
                
        if st.checkbox('Interpret'):
            st.subheader('Interpretation of Model')
            if model_choice == 'Random Forest':
                loaded_model = load_model("Random_Forest_Model.pkl")
                
            df1 = pd.read_csv("Datasets/balanced_data.csv")
            x = df1[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
            feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
            class_names = ['Die(1)', 'Live(2)']
            explainer = lime.lime_tabular.LimeTabularExplainer(x.values, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
                
            # The lime explainer
            exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba, num_features=14)
            exp.show_in_notebook(show_table=True)
            fig = exp.as_pyplot_figure()
            st.pyplot()


if __name__ == '__main__':
    main()

