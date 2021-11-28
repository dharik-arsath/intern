#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Add the csv file into cassandra database
# Get the data from cassandra to python pandas
# split the dataset into train, valid and test (Best practice)
# Analyze the data
# make a report of analysis
# use mlflow if needed and 
# build a model
# Analyze the model and make a report

# use lime and explain the model.
# predict
# use streamlit for deployment. 
# deploy


# In[189]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pymongo
from sklearn.model_selection import train_test_split

from ipywidgets import widgets
from IPython.display import display


from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import math

import lightgbm
from pymongo import MongoClient


# In[ ]:





# In[190]:


# data = pd.read_csv('/home/arsath/ineuron_internship/adult.csv')
# data.columns = data.columns.str.replace('-','_')
# data.head()

# cols = data.select_dtypes(['object', 'category'])
# for x in cols:
#     data[x] = (
#         data[x].str.strip()
#         .astype('category')
#     )
    
# data.drop('education_num', axis=1, inplace=True)
# school_students = {'1st-4th' : 'some-school', 
#                    '5th-6th' : 'some-school',
#                    '7th-8th' : 'some-school',
#                    '9th'     : 'some-school', 
#                    '10th'    : 'some-school',
#                    '11th'    : 'some-school',
#                    '12th'    : 'some-school'
#                   }
# data.education.replace(school_students, inplace=True)

# marriage_correction = {'Separated': 'Married_lost',
#                        'Widowed' : 'Married_lost', 
#                        'Married-spouse-absent' : 'Married_lost',
#                        'Divorced' : 'Married_lost'
#                        }
# data.marital_status.replace(marriage_correction, inplace=True)

# data.workclass.replace({'Without-pay':'Never-worked'},inplace=True)

# data.race.replace(to_replace=['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], value='Other', inplace=True)
# data['total_gain'] = data.capital_gain.values - data.capital_loss.values
# data.drop(['capital_gain', 'capital_loss'], axis=1, inplace=True)
# data.workclass.replace(['State-gov', 'Federal-gov', 'Local-gov'], 'government', inplace=True)
# data.education.replace(['Assoc-voc', 'Assoc-acdm'], 'Assoc', inplace=True)


#data_dict = data.to_dict('records')
#collection.insert_many(data_dict)


# In[191]:


def connect_to_db():
    client = MongoClient()
    db = client.test_database
    collection = db['adult']
    return collection


# In[192]:


collection = connect_to_db()
data = pd.DataFrame(collection.find()).drop(['_id'],axis=1)
data


# In[193]:


def make_test_data():

    X = data.drop('salary', axis=1)
    y = data.salary

    _, X_test, _, y_test = train_test_split(X,y,test_size=0.07)

    pd.concat([X_test, y_test], axis=1).to_csv('/home/arsath/ineuron_internship/test_data.csv',index=False)
    data.drop(X_test.index, inplace=True)


# In[194]:


make_test_data()


# In[9]:


px.histogram(data_frame=data, x='occupation', color='salary')


# In[150]:





# In[7]:


occ_wid = widgets.Dropdown(options=data.occupation.unique(), 
                 value='Prof-specialty', 
                 description = 'Occupation: ')


# In[8]:


out = widgets.Output(layout= widgets.Layout(height='600px'))
def occ_wid_handler(val):
    with out:
        out.clear_output()
        to_plot = data[data.occupation.values == val.new]
        pie_plot = to_plot.value_counts('salary')
        display(px.pie(data_frame=data, 
                       names=pie_plot.index, 
                       values=pie_plot.values,
                       title = f'Salary percentage of {val.new}')
               )        
occ_wid.observe(occ_wid_handler, names='value')


# In[9]:


display(occ_wid)
display(out)


# In[10]:


workclass_wid = widgets.Dropdown(options=data.workclass.unique(), 
                 value='government', 
                 description = 'Workclass: ')


# In[11]:


out_workclass = widgets.Output(layout= widgets.Layout(height='600px'))
def workclass_wid_handler(val):
    with out_workclass:
        out_workclass.clear_output()
        pie_plot = data[data.workclass.values == val.new].value_counts('salary')
        display(px.pie(data_frame=data, 
                       names=pie_plot.index, 
                       values=pie_plot.values,
                       title = f'Salary percentage of {val.new}')
               )        
workclass_wid.observe(workclass_wid_handler, names='value')


# In[12]:


display(workclass_wid)
display(out_workclass)


# In[9]:


pd.options.plotting.backend = "plotly" 


# ## We have tried multiple bins and 4 suits the best  to split

# In[10]:


data.age.hist(nbins=4)


# In[5]:


data['age_bin'] = pd.cut(data.age.values, bins=[17,25, 40, 60, 100])


# ## Total Gain highly influencing the salary. Higher the total gain, greater the salary.

# In[78]:


px.box(data_frame=data,x='salary', y='total_gain',log_y=True)


# In[81]:


data.loc[data.total_gain > 2000, 'salary'].value_counts()


# In[105]:


data.hours_per_week.hist(nbins=5)


# In[100]:


data.hours_per_week.describe()


# In[6]:


data['hours_per_week_bin'] = pd.cut(
    data.hours_per_week, bins=[1, 20, 40, 60, 80, 100]
)


# In[ ]:





# In[ ]:





# ## HoursPerWeek and age combined together makes intresting analysis.

# In[92]:


px.scatter(data_frame=data, x='age', y='hours_per_week', color='salary')


# In[94]:


data.loc[(data.age.values > 25) & (data.age.values < 60), 'salary'].value_counts()


# In[95]:


data.salary.value_counts()


# ## Out of 7841, 7075 people are those whose age is greater than 25 and less than 60

# In[ ]:





# In[15]:


pd.crosstab(data.salary, data.age_bin,data.hours_per_week,aggfunc='mean')


# In[163]:


pd.crosstab(data.salary, data.hours_per_week_bin)


# In[ ]:





# In[228]:


data.head()


# In[234]:


data.plot('total_gain',color='salary', kind='hist', nbins=5)


# In[237]:


data.total_gain.describe()


# In[7]:


data['total_gain_bin'] = pd.cut(
    data.total_gain, bins=[
        data.total_gain.min()-1,0, 50000, data.total_gain.max()
    ]
)


# In[ ]:





# In[195]:


from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[196]:


def create_model_data(model_data, to_drop=None):
    if to_drop is not None:
        model_data = model_data.drop(to_drop,axis=1)

    return model_data


# In[197]:


def get_train_valid(model_data, target='salary'):
    X = model_data.drop([target], axis=1)
    y = model_data.salary

    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.1)
    return X_train, X_valid, y_train, y_valid


# In[198]:


from sklearn.preprocessing import LabelEncoder

def fit_encode_target(y_train):    
    out_enc = LabelEncoder()
    y_train = out_enc.fit_transform(y_train)
    return y_train, out_enc


# In[199]:


def predict_encode_target(y_test, encoder):
    return encoder.transform(y_test)


# In[200]:


def fit_transformer_pipeline(trans_data):
    obj_data = trans_data.select_dtypes(
        include = ['category', 'object']).columns
    
    num_data = trans_data.select_dtypes(
        include = ['int', 'float']).columns
    
    colt = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), num_data),
        ('ordenc', OrdinalEncoder(), obj_data)
    ])

    trans_data = pd.DataFrame(colt.fit_transform(trans_data),
                           columns=np.concatenate([num_data, obj_data]))


    return trans_data, colt


# In[201]:


def predict_transformer_pipeline(trans_data, encoder):
    return pd.DataFrame(encoder.transform(trans_data), 
                        columns=trans_data.columns)


# In[202]:


def fit_model(X_train, X_valid, y_train, y_valid):
    
    train_data = lightgbm.Dataset(data=X_train, label=y_train)
    valid_data = lightgbm.Dataset(data=X_valid, label=y_valid, 
                                  reference=train_data)
    
    SEARCH_PARAMS = {
        'feature_fraction': 0.7,
        'subsample': 0.7,
        'num_leaves' : 30,
        'n_estimators' : 500
                }


    FIXED_PARAMS={'objective': 'binary',
             'metric': 'auc',
             'is_unbalance':True,
             'bagging_freq':5,
             'num_boost_round':300
             }


    params = {
        'metric':FIXED_PARAMS['metric'],
        'objective':FIXED_PARAMS['objective'],
        **SEARCH_PARAMS
         }

    lgb = lightgbm.train(
        params, train_data,
        valid_sets=[valid_data],
        callbacks = [lightgbm.early_stopping(50)],
        num_boost_round=FIXED_PARAMS['num_boost_round'],
        valid_names=['valid']
    )
    
    return lgb


# In[203]:


def train_full_pipeline(data, to_drop=None):
    model_data = create_model_data(data, to_drop)
    X_train, X_valid, y_train, y_valid = get_train_valid(
        model_data, target='salary'
    )
    
    y_train, enc = fit_encode_target(y_train)
    y_valid = predict_encode_target(y_valid, encoder = enc)
    model_train_data, col_t = fit_transformer_pipeline(X_train)
    model_valid_data = predict_transformer_pipeline(X_valid, col_t)
    
    model = fit_model(model_train_data, model_valid_data, 
                      y_train, y_valid)
    return model, enc, col_t


# In[204]:


def get_data_label(test_dataset, target_col):
    """
    Separates target variable from independent variable
    """
    test_data = test_dataset.drop(target_col, axis=1)
    test_label = test_dataset.loc[:,target_col]
    return test_data, test_label


# In[205]:


def predict_model(model, X_test, threshold = 0.5):
    pred = (model.predict(X_test) > threshold)  * 1
    return pred


# In[206]:


def get_metrics(y_true, y_pred, func, **kwargs):
    return func(y_true, y_pred, **kwargs)


# In[207]:


def predict_full_pipeline(data, target_encoder, col_transformer, model,
                          to_drop=None):
    model_data = create_model_data(data, to_drop)
    test_data, test_label = get_data_label(
        model_data, target_col='salary'
    )
    test_label = predict_encode_target(test_label, encoder=target_encoder)
    test_data = predict_transformer_pipeline(
        test_data, encoder=col_transformer
    )

    pred = predict_model(model, test_data, threshold=0.5)
    return pred, test_label


# In[208]:


model, enc, col_t = train_full_pipeline(data, to_drop=['country'])


# In[217]:


new_test_data = pd.read_csv('/home/arsath/ineuron_internship/test_data.csv')

prediction,act_label = predict_full_pipeline(new_test_data, enc, col_t, model,
                      to_drop=['country'])


# In[218]:


from sklearn.metrics import classification_report
print(get_metrics(act_label, prediction, classification_report))


# In[219]:


new_test_data = create_model_data(new_test_data, to_drop = ['country'])
test_data, test_label = get_data_label(new_test_data, target_col='salary')


# In[220]:


import shap


# In[221]:


explainer = shap.TreeExplainer(model)

test_data = predict_transformer_pipeline(test_data, encoder=col_t)
shap_values = explainer.shap_values(test_data)


# In[97]:


shap.initjs()


# In[222]:


shap.summary_plot(shap_values, test_data)


# In[ ]:





# In[183]:


model.save_model('model')


# In[ ]:


lightgbm.Booster(model_file = '/home/arsath/ineuron_internship/model')


# In[ ]:





# In[186]:


import joblib

joblib.dump(col_t, 'transformer')


# In[187]:


joblib.dump(enc, 'encoder')


# In[188]:


joblib.load('transformer')


# In[169]:


# TODO 

# Ask question from streamlit and provide answers through your analysis

