import pandas as pd

# This object imports training and test .csv files

class Loader: 
    
    def __init__(self, trainPath, testPath):
        Loader.trainPath = trainPath
        Loader.testPath = testPath
        print('Path selected\n')
    
    def importData(self):
        # Import files
        train = pd.read_csv(Loader.trainPath, index_col="INDEX")
        test = pd.read_csv(Loader.testPath, index_col="INDEX")
        print('Succesfully loaded data\n')
        return train, test