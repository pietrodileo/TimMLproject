# Tim ML Engineer project
This is a simple binary classification project involving machine learning. The dataset is related to something like auto insurance and contains more than 8000 instances (8161). There are 25 features, both numerical and categorical, and the target variable is "TARGET_FLAG". The dataset is unbalanced (6008 instances for class 0, 2153 instances for class 1), so it required some preprocessing before training the models.

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
The project can run with just a launch script of the main file. The project was developed on VS Code with Python 3.1
Please follow these commands in powershell:

if you want to use a virtual environment, in the current folder do:
1. Create the V.E.
python -m venv vEnv
2. Activate the V.E.
.\venv\Scripts\Activate.ps1
If it is impossible to activate the environment try writing before:
Set-ExecutionPolicy Unrestricted -Scope Process
(type deactivate to deactivate the environment)

Execute code:
3. Install required libraries
pip install -r .\requirements.txt
4. To run the code just do:
python .\main.py  

After this, Streamlit app is automatically launched. If you don't want to launch the app please comment line 34 ("webapp(mdl)") in "main.py".

5. CTRL + C to close the app.
Please close the app before closing the browser.

6. After one run the model and the data were saved in f_app folder, so you can directly run the web app without running main.py another time.
To do this write:
streamlit run .\\f_app\\app.py --server.port 9800
Port 9800 was choosen in order to avoid errors.