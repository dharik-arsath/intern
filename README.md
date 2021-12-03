# intern

### Adult.csv 
  The actual dataset which was given for the given problem, we have further divided this dataset into training and testing for the purpose of avoiding data leakage which is one of the major problem in the field of data science. 

### Train_data.csv 
  Training dataset which was derived from actual data (Adult.csv) and saved into the disk for the purpose of Training.
  It is always recommended to split the data into training and testing and store both the data into disks inorder to avoid overfitting.

### Test_data.csv
  Test dataset which was derived from actual data (Adult.csv) and saved into disk for the purpose of Testing. Model will be trained on the Training Dataset and once the model is finalized this test dataset will be used for the purpose of testing the generalization of the model.

### App_code.ipynb 
  Jupyter-notebook file for the data and the model. The actual code from reading to saving all the models,encoders and transformers have been done in this notebook file.

### Viz.py 
  Python file which is a code for streamlit app. Streamlit turns data scripts into shareable web apps in minutes.
All in Python. All for free. No front‑end experience required. This makes data scientists focus more on the data and the model not on the javascript or anyother web related languages. Everything is done in python and most importantly streamlit makes data scientists task much easier and prettier as well.


### Encoder: 
  Label Encoder that was used in the training dataset (Best practice is to save the encoder into disk and reuse with test set). It is always best practise to store the encoders, transformers and model into disk for resuing later.

### Transformer: 
  Column Transformer which is used in the training dataset and it is required to save the transformer into disk for reusing later in the testset.

### Model
  LightGBM model file for the Adult Census Income Prediction which was used in this dataset, again it is best practise to save the model into disk for resusing later. This is the actual model which is deployed in the streamlit platform, we have used lightgbm in this project which is scalable and more accurate in the modern era of machine learning algorithms (mostly).

### Deployment Link
  Link for the final Streamlit app which was created and deployed in the streamlit and streamlit share.





