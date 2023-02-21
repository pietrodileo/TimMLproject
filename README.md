# Tim ML Engineer project
This is a simple binary classification project involving machine learning. The dataset is related to something like auto insurance and contains more than 8000 instances (8161). There are 25 features, both numerical and categorical, and the target variable is "TARGET_FLAG". The dataset is unbalanced, so it required some preprocessing before training the models.

## Project Workflow
The project follows the following passages: 
1. Data are loaded from a .csv file
2. The original data is preprocessed in order to obtain a dataset suitable for training.
3. The dataset is balanced through three different resampling techniques, which are then compared.
4. Four different classification models are trained using both the original and resampled datasets. The results are compared to find the best model/resample combination. 
5. The best model underwent a simple hyperparameters tuning and then evaluated on a separate test set.
6. Finally the model is used to make prediction of another dataset. 
7. At the end of the process the script launches a simple Streamlit web app, which can be used to do some exploration of the data and also to predict the class for a new instance created interactively by the user. 

## Commands
The project can run with just a launch script of the main file. 
Please follow these commands:

