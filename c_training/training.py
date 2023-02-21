from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# This object compares different training algorithms and selects the best one

class Trainer: 

    def __init__(self):
        print('Trainer object was created\n')

    def trainMdl(mdl, X_train, y_train, X_test, y_test):
        # train a model
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        return f1_score(y_test, y_pred)

    def compareClassifiers(self, df, target):
        # compare performance of different classification algorithms
        trainData = df.copy()
        # extract numerical values and label
        if target in trainData.columns:
            ClassVector = trainData[target]
            trainData.drop(target, axis=1, inplace=True)

        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(trainData, ClassVector, test_size = 0.2, random_state = 0)
    
        # train a classification model 
        svm_rbf = svm.SVC()
        dt = DecisionTreeClassifier()
        nn = MLPClassifier(solver='adam', max_iter=800, hidden_layer_sizes=(10, 2), random_state=1)
        rf = RandomForestClassifier()

        # fit the models
        f1svm = Trainer.trainMdl(svm_rbf,X_train,y_train,X_test,y_test)
        f1dt = Trainer.trainMdl(dt,X_train,y_train,X_test,y_test)
        f1nn = Trainer.trainMdl(nn,X_train,y_train,X_test,y_test)
        f1rf = Trainer.trainMdl(rf,X_train,y_train,X_test,y_test)
        
        # export the results
        f1Df = pd.DataFrame([[f1svm, f1dt, f1rf, f1nn]], columns=['SVM','DT','RF','NN'])
        return f1Df

    def compare(self, f1Ori, f1under, f1over, f1smote):
        # compare results obtained with different classifiers and find the best model
        f1df = pd.concat([f1Ori, f1under, f1over, f1smote])
        f1df.index = ['Original Data','UnderSample','OverSample','SMOTE']
        print(f1df)
        # Find the best combination
        bestf1 = f1df.to_numpy().max()
        positionBest = f1df == bestf1
        # find the columns containing True 
        seriesObj = positionBest.any()
        # find best model
        bestMdl = seriesObj.index[seriesObj == True].tolist() 
        bestPreproc = positionBest.index[positionBest[bestMdl[0]] == True].tolist()
        print('\n' + bestMdl[0] + ' classifier with ' +  bestPreproc[0] + ' preprocessing achieved the best F1-Score of ' + str(bestf1))

    def tuneBest(self, bestDataset, target):
        # Simple hyperparameter tuning of a RF model (considered the best classifier by the results)
        print('\nModel tuning...\n')
        # set grid for CV Grid Search
        param_grid = { 
            'n_estimators': [100, 200, 300],
        }
        trainData = bestDataset.copy()
        # extract numerical values and label
        if target in trainData.columns:
            ClassVector = trainData[target]
            trainData.drop(target, axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(trainData, ClassVector, test_size = 0.2, random_state = 0)
        # Perform grid search
        CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='f1_macro')
        CV_rfc.fit(X_train, y_train)
        # print best results
        print(CV_rfc.best_params_)
        print('Mean CV results: ' + str(CV_rfc.cv_results_['mean_test_score']))
        
        print(trainData.columns)
        # Train the tuned model
        mdl = RandomForestClassifier(n_estimators=CV_rfc.best_params_['n_estimators'])
        mdl.fit(X_train, y_train) 

        return mdl, X_train, X_test, y_train, y_test
