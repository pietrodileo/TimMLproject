# TimDSproject
This is a simple binary classification project involving machine learning. 
The dataset is related to something like auto insurance and contains more than 8000 instances (8161). There are 25 features, both numerical and categorical, and the target variable is "TARGET_FLAG". The dataset is unbalanced (6008 instances for class 0, 2153 instances for class 1), so it required some preprocessing before training the models.

## Project Workflow
The project follows the following passages: 
1. Data are loaded from a .csv file
2. The original data is preprocessed in order to obtain a dataset suitable for training.
3. The dataset is balanced through three different resampling techniques, which are then compared.
4. Four different classification models are trained using both the original and resampled datasets. The results are compared to find the best model/resample combination. 
5. The best model underwent a simple hyperparameters tuning and then evaluated on a separate test set.
6. Finally the model is used to make prediction of another dataset and the results are saved in "predictionResults.csv". 

## Commands
The project can run with just a launch script of the main file. The project was developed on VS Code with Python 3.1
Please follow these commands in powershell:

if you want to use a virtual environment, in the current folder do:
1. Create the V.E.

python -m venv vEnv

2. Activate the V.E.

.\venv\Scripts\Activate.ps1

If it is impossible to activate the environment try writing the following before the activation:
Set-ExecutionPolicy Unrestricted -Scope Process
(type deactivate to deactivate the environment)

Execute code:
3. Install required libraries

pip install -r .\requirements.txt

4. Just simply run 'main.ipynb'

