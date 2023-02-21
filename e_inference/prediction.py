import pandas as pd

# This object predicts the class of new data and export results

class Predictor:
    def __init__(self):
        print('Predictor Object was created\n')

    def prediction(self, mdl, test):
        # select index
        idx = test.index.to_list()
        # predict class
        ypred_test = pd.DataFrame(mdl.predict(test))
        ypred_test.columns = ['predictions']
        # predict probabilities
        y_pred_prob_test = mdl.predict_proba(test)
        # select probability for the selected class
        max_probab = pd.DataFrame(y_pred_prob_test.max(axis=1))
        max_probab.columns = ['probabilities']
        idx = pd.DataFrame(idx)
        idx.columns = ['idx']
        # create a data frame
        predResults = pd.concat([idx, ypred_test, max_probab], axis=1)
        predResults.set_index('idx',inplace = True)
        # export results
        predResults.to_csv('.\\e_inference\\predictionResults.csv')  
        print(predResults)
        return predResults