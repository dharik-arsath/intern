import streamlit as st
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import numpy as np



def get_data_model():

	file = st.file_uploader("Choose a File ", type=["csv"])
	if file is not None:
		file = pd.read_csv(file)

		model = lgb.Booster(model_file = 'model')

		return file ,model
	else:
		return None, None


@st.cache
def preprocess(data):
	 data.columns = data.columns.str.replace('-','_')


	 cols = data.select_dtypes(['object']).columns
	 for x in cols:
	 	data[x] = (
	        data[x].str.strip()
	        .astype('category')
	    )
	    
	 data.drop('education_num', axis=1, inplace=True)
	 school_students = {'1st-4th' : 'some-school', 
	                   '5th-6th' : 'some-school',
	                   '7th-8th' : 'some-school',
	                   '9th'     : 'some-school', 
	                   '10th'    : 'some-school',
	                   '11th'    : 'some-school',
	                   '12th'    : 'some-school'
	                  }
	 data.education.replace(school_students, inplace=True)

	 marriage_correction = {'Separated': 'Married_lost',
	                       'Widowed' : 'Married_lost', 
	                       'Married-spouse-absent' : 'Married_lost',
	                       'Divorced' : 'Married_lost'
	                       }
	 data.marital_status.replace(marriage_correction, inplace=True)

	 data.workclass.replace({'Without-pay':'Never-worked'},inplace=True)

	 data.race.replace(to_replace=['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], value='Other', inplace=True)
	 data['total_gain'] = data.capital_gain.values - data.capital_loss.values
	 data.drop(['capital_gain', 'capital_loss'], axis=1, inplace=True)
	 data.workclass.replace(['State-gov', 'Federal-gov', 'Local-gov'], 'government', inplace=True)
	 data.education.replace(['Assoc-voc', 'Assoc-acdm'], 'Assoc', inplace=True)
	 data.drop('country', axis=1, inplace=True)


	 return data


def infer(model, file):

	col_t = load('transformer')
	enc = load('encoder')


	test_data = file.drop('salary', axis=1)
	test_label = file.salary

	st.write(test_data)

	num_data_cols = test_data.select_dtypes(include=['int', 'float']).columns
	obj_data_cols = test_data.select_dtypes(include=['object', 'category']).columns
	test_data = pd.DataFrame(col_t.transform(test_data), columns=np.concatenate([num_data_cols, obj_data_cols]))
	test_label =  enc.transform(test_label)


	threshold =0.5
	prediction = (model.predict(test_data) > threshold)  * 1

	st.write(classification_report(prediction, test_label))

	st.write(confusion_matrix(prediction , test_label))

	return test_data, col_t

def dashboard(model, test_data, col_t):
	model.params['objective'] = 'binary' # or whatever is appropriate
	explainer = shap.TreeExplainer(model)


	shap_values = explainer.shap_values(test_data)

	fig, ax = plt.subplots()

	ax = shap.summary_plot(shap_values, test_data)
	st.pyplot(fig)



if __name__ == '__main__':

	file, model = get_data_model()
	if file is not None:
		file = preprocess(file)
		test_data, col_t = infer(model, file)
		dashboard(model, test_data, col_t)










