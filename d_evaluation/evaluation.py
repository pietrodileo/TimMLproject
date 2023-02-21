from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# This object evaluates classifier performance

class Evaluator:
    def __init__(self):
        print('Evaluator Object was created')
    
    def evaluateMdl(self, mdl,X_test,y_test):
        y_pred=mdl.predict(X_test)
        print("Accuracy on test data: ",accuracy_score(y_test,y_pred))
        print("F1-Score on test data: ",f1_score(y_test,y_pred))
        # Print the Confusion Matrix
        print('\nConfusion Matrix\n')
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return y_pred, cm