from a_ingestion.load_data import Loader
from b_process.preprocess import Preprocessor
from b_process.balanceDataset import Balancer
from c_training.training import Trainer
from d_evaluation.evaluation import Evaluator
from e_inference.prediction import Predictor
import os 
import pickle

k = 13

def run():
    print('\n'+'*'*k + ' Process starts ' + '*'*k)
    # Loading data
    trainOri, testOri = load_data()
    # Preprocess data
    train_p, test_p, traincleaned = preprocessing(trainOri, testOri)
    # save data for streamlit
    traincleaned.to_csv('.\\f_app\\traincleaned.csv')  
    # Balance data
    target = 'TARGET_FLAG'
    # balancer contains three resampled dataset: train_under, train_over and train_smote
    balancer = balancingDataset(train_p, target) 
    # Training and select the best performing model\library\advanced-features\cli
    mdl, X_train, X_test, y_train, y_test = training(train_p,target,balancer)
    # save the model for the web app
    pickle.dump(mdl, open('.\\f_app\\mdl.pickle', "wb"))
    # Evaluate test set
    y_pred, cm = evaluation(mdl,X_test,y_test)
    # Make prediction
    predResults = prediction(mdl, test_p)
    print('\n' + '*'*k + ' Process ends ' + '*'*k)
    # Run Streamlit web app
    webapp(mdl)

def load_data():
    print('\n'+'*'*k + ' A. Loading data ' + '*'*k)
    # set path
    trainPath = 'a_ingestion/train_auto.csv'
    testPath = 'a_ingestion/test_auto.csv'
    # define a Loader object
    loadData = Loader(trainPath,testPath)
    # load data from .csv file
    trainOri, testOri = loadData.importData()
    return trainOri, testOri

def preprocessing(trainOri, testOri):
    print('\n'+'*'*k + ' B. Preprocessing' + '*'*k)
    # define a Preprocessor object
    preprocess = Preprocessor(trainOri, testOri)

    # Remove useless columns from the dataframes
    preprocess.dropVar(preprocess.train, ['TARGET_AMT'])
    preprocess.dropVar(preprocess.test, ['TARGET_FLAG','TARGET_AMT'])

    # Explore variables
    preprocess.showVarType(preprocess.train)

    # Some numerical variables like INCOME were interpreted as numerical (ex. '$200'). 
    # We convert them from string to float
    print('\nConverting categorical variables to numerical...')
    cat2numVars = ["INCOME","HOME_VAL","BLUEBOOK","OLDCLAIM"]
    preprocess.train = preprocess.str2int(preprocess.train, cat2numVars)
    preprocess.test = preprocess.str2int(preprocess.test, cat2numVars)

    # Fill missing values
    print('\nShow missing values: ')
    print(preprocess.train.isna().sum())    
    print('\nFill missing values...')
    preprocess.fillMissing(preprocess.train,preprocess.test,'TARGET_FLAG')  
    print('\nCheck if missing values were filled:')
    print(preprocess.train.isna().sum())
    
    # Cleaned data for web app exploration
    traincleaned = preprocess.train.copy()

    # Scale data
    print('\nStandardizing numerical variables...')
    [preprocess.train,preprocess.test] = preprocess.scaling(preprocess.train, preprocess.test, ['PARENT1', 'MSTATUS', 'RED_CAR', 'REVOKED',
                                                                          'SEX','CAR_USE','URBANICITY','JOB','CAR_TYPE','EDUCATION'],
                                                                          target='TARGET_FLAG')

    # convert categorical variables to numerical
    print('\nConverting categorical variables into dummies...')
    preprocess.train = preprocess.convertCategorical(preprocess.train)
    preprocess.test = preprocess.convertCategorical(preprocess.test)

    return preprocess.train,preprocess.test, traincleaned

def balancingDataset(train, target):
    print('\n'+'*'*k + ' C. Balancing Dataset ' + '*'*k)
    # define a balancer object
    balancer = Balancer()
    # show istances belonging to each class
    balancer.describeSet(train,target)
    # resample dataset
    balancer.train_under,balancer.train_over,balancer.train_smote = balancer.resample(train,target)
    return balancer

def training(train,target,balancer):
    print('\n'+'*'*k + ' D. Training Models ' + '*'*k)
    trainer = Trainer()
    # compare several classification algorithms and resampling methods
    print('\nTraining...')
    f1Ori = trainer.compareClassifiers(train, target)
    f1under = trainer.compareClassifiers(balancer.train_under,target)
    f1over = trainer.compareClassifiers(balancer.train_over,target)
    f1smote = trainer.compareClassifiers(balancer.train_smote,target)
    #compare results obtained with different classifiers and resampling methods
    trainer.compare(f1Ori, f1under, f1over, f1smote)
    # Show best dataset and training algorithm
    bestDataset = balancer.train_over
    mdl, X_train, X_test, y_train, y_test = trainer.tuneBest(bestDataset, target)
    return mdl, X_train, X_test, y_train, y_test

def evaluation(mdl,X_test,y_test):
    # evaluate classifier performance
    print('\n'+'*'*k + ' E. Model Evaluation on Test set ' + '*'*k)
    evaluator = Evaluator()
    print('\nEvaluating model performance...')
    y_pred, cm = evaluator.evaluateMdl(mdl,X_test,y_test)
    return y_pred, cm

def prediction(mdl,test):
    # predict the class of new data
    print('\n'+'*'*k + ' F. Prediction ' + '*'*k)
    print('\nPredicting...')
    predictor = Predictor()
    # Predict result
    predResults = predictor.prediction(mdl, test)
    return predResults

def webapp(mdl):
    # Run a webapp with streamlit
    print('Loading a web app page')
    # Run webapp on port 9800
    os.system('streamlit run .\\f_app\\app.py --server.port 9800')

if __name__ == '__main__':
    run()