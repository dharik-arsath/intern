# intern

Adult.csv : The actual dataset which was given for the given problem

Train_data.csv : Training dataset which was derived from actual data (Adult.csv) and saved into the disk for the purpose of Training

Test_data.csv : Test dataset which was derived from actual data (Adult.csv) and saved into disk for the purpose of Testing

App_code.ipynb : Jupyter-notebook file for the data and the model.

Viz.py : Python file which is a code for streamlit app.

encoder: Label Encoder that was used in the training dataset (Best practice is to save the encoder into disk and reuse with test set)

transformer: Column Transformer which is used in the training dataset and it is required to save the transformer into disk for reusing later in the testset.

model : LightGBM model file for the Adult Census Income Prediction which was used in this dataset, again it is best practise to save the model into disk for resusing later.

Deployment Link: Link for the final Streamlit app which was created and deployed in the streamlit and streamlit share.





