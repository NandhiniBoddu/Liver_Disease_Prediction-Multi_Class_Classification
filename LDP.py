import pandas as pd 
import numpy as np
import streamlit as st 
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
from sklearn.ensemble import RandomForestClassifier

st.title('LIVER DISEASE PREDICTION')
st.image('https://www.ddw-online.com/wp-content/uploads/2022/10/Shutterstock_1906753141-1.jpg')
st.sidebar.header('User Input Parameters')

def user_input_features():
    Age = st.sidebar.number_input("Insert the Age")
    Sex = st.sidebar.selectbox('Gender (Male=1,Female=0)',('1','0'))
    Albumin = st.sidebar.number_input("Insert the Albumin level")
    Alkaline_phosphatase = st.sidebar.number_input("Insert the Alkaline Phosphatase level")
    Alanine_aminotransferase = st.sidebar.number_input("Insert the Alanine Aminotransferase level")
    Aspartate_aminotransferase = st.sidebar.number_input("Insert the Aspartate Aminotransferase level")
    Bilirubin = st.sidebar.number_input("Insert the Bilirubin level")
    Cholinesterase = st.sidebar.number_input("Insert the Cholinesterase level")
    Cholesterol = st.sidebar.number_input("Insert the Cholesterol level")
    Creatinina = st.sidebar.number_input("Insert the Creatinina level")
    Gamma_glutamyl_transferase = st.sidebar.number_input("Insert the Gamma Glutamyl Transferase level")
    Protein = st.sidebar.number_input("Insert the Protein level")

    df = {'age':Age, 'sex':Sex, 'albumin':Albumin,
        'alkaline_phosphatase':Alkaline_phosphatase, 
        'alanine_aminotransferase':Alanine_aminotransferase,
        'aspartate_aminotransferase':Aspartate_aminotransferase,
        'bilirubin':Bilirubin, 'cholinesterase':Cholinesterase,
        'cholesterol':Cholesterol, 'creatinina':Creatinina,
        'gamma_glutamyl_transferase':Gamma_glutamyl_transferase,
        'protein':Protein}

    features = pd.DataFrame(df,index=[0])
    return features

df1 = user_input_features()
st.subheader('User Input Parameters')
st.write(df1)

data = pd.read_csv('project-data.csv', delimiter=';')

data.rename(columns = {'protein   ' : 'protein', 'gamma_glutamyl_transferase ' : 'gamma_glutamyl_transferase'}, inplace = True)
data['protein'] = pd.to_numeric(data['protein'], errors='coerce')


median=data.median()
data=data.fillna(median) 

data['sex'] = np.where(data['sex'].str.contains("m"), 1, 0) # 1=m , 0=f  
data['category']= label_encoder.fit_transform(data['category']) 

X = data[["age","sex","albumin","alkaline_phosphatase","alanine_aminotransferase","aspartate_aminotransferase","bilirubin","cholinesterase","cholesterol","creatinina","gamma_glutamyl_transferase","protein"]]
Y = data[["category"]]


clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df1)
prediction_probability = clf.predict_proba(df1) 

st.subheader('Predicted Result')
st.write('Cirrhosis' if prediction==0 else '')
st.write('Fibrosis' if prediction==1 else '')
st.write('Hepatitis' if prediction==2 else '')
st.write('No Disease' if prediction==3 else '')
st.write('Suspect Disease' if prediction==4 else '')

st.subheader('Prediction Probability')
st.text('''O--> Cirrhosis
1--> Fibrosis
2--> Hepatitis
3--> No Disease
4--> Suspect Disease''')
st.write(prediction_probability)


st.title("**About**")
st.write("**This Streamlit application is using Random Forest Machine Learning model developed by:**")
st.write("***Vishnu***",",","***Nandhini***")
st.write("***Naveen Kumar***",",","***Divya***")
st.write("***Pradeep Kumar***",",","***Akshay***")

st.header("Guided by:")
st.write("**Varun Vennelaganti**")









