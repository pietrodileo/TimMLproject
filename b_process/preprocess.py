#import sys
#sys.path.append("..")
#from Folder.a_ingestion import train, test
import pandas as pd
from sklearn.preprocessing import StandardScaler

# This object performs several preprocessing of the data set

class Preprocessor: 

    def __init__(self, train, test):
        Preprocessor.train = train
        Preprocessor.test = test
        print('Preprocessor object was created\n')

    def dropVar(self, df, vars):
        # Remove a variable from the dataset
        df.drop(vars, axis=1, inplace = True)
        print(str(vars) + ' removed')

    def showVarType(self, df):
        # Obtain a pandas series containing features data types
        print("\nExploring data types...")
        result = pd.DataFrame(df.dtypes, columns=['Type'])
        result.reset_index(inplace=True, names=['Features'])
        # Show data types
        print(result.groupby(['Type']).count()) 
        # Print Categorical variables
        print('\nThe following categorical variables were founded:')
        catVars = result.loc[result['Type'] == "object"]
        print(catVars['Features'])

    def str2int(self, df, variables):
        # Define a function that extract the numerical value from a string like '$200' and convert it to a float
        c = 0  # c is an index that tells if a conversion was made
        newdf = df.copy()
        for var in variables:
            if df[var].dtypes == 'object': 
                # if it was a string, convert variable to float 
                newdf[var] = df[var].apply(lambda x: float(x.split("$")[1].replace(",","")) if type(x) == str else x)
                print("Variable " + var + " was converted!")
                c = 1
            else: # if variable was converted already its tipe will be float64
                print("Variable " + var + " is not categorical")
        # Return an output  
        if c == 1:
            return newdf
        else: 
            return df

    def scaling(self, traindf, testdf, catVars, target):
        # setup scaler
        scaler = StandardScaler()
        train_s = traindf.copy()
        test_s = testdf.copy()
        # remove flag from the df
        catVarsTest = catVars.copy()
        # remove both target variables because in the test data they are not provided
        catVars = catVars + [target]
        train_s.drop(catVars, axis=1, inplace=True)
        test_s.drop(catVarsTest, axis=1, inplace=True)
        # training set scaling
        train_s = pd.DataFrame(scaler.fit_transform(train_s), 
                               columns=train_s.columns, index=train_s.index)
        # test set scaling
        test_s = pd.DataFrame(scaler.transform(test_s),
                          columns=test_s.columns, index=test_s.index)
        # re-add categorical variables
        train_s = train_s.join(traindf[catVars])
        test_s = test_s.join(testdf[catVarsTest])
        return train_s, test_s

    def fillMissing(self,train_s,test_s,target):  
        # fill NaN with mean and mode
        vars = train_s.columns.drop(target)
        for var in vars: 
            # TRAIN
            if any(train_s[var].isna()):
                if train_s[var].dtypes == 'float64':
                    # substitute NaN with mean
                    train_s[var].fillna(value=train_s[var].mean(), 
                                        inplace=True)
                elif train_s[var].dtypes == 'object':
                    # substitute NaN with mode
                    train_s[var].fillna(value=train_s[var].mode()[0], 
                                        inplace=True)        
            # TEST
            if any(test_s[var].isna()):
                if test_s[var].dtypes == 'float64':
                    # substitute NaN with mean
                    test_s[var].fillna(value=train_s[var].mean(), 
                                       inplace=True)
                elif test_s[var].dtypes == 'object':
                    # substitute NaN with mode
                    test_s[var].fillna(value=train_s[var].mode()[0],
                                       inplace=True)

    def cat2int(self, df, variables):
        # Convert categorical variables to numerical
        c = 0 # c is an index that tell if a conversion was made
        # Create a copy of the original dataframe to avoid warnings
        newdf = df.copy()
        firstidx = df.index[0]
        for var in variables:
            if type(df[var][firstidx]) == str:
                newdf[var] = df[var].apply(lambda x: 1 if 'yes' in x.lower() else 0)
                print("Variable " + var + " was converted!")
                c = 1
            else:
                # if we run the code again it will
                print("Variable " + var + " is not categorical")
        # Return an output  
        if c == 1:
            return newdf
        else: 
            return df
    
    def convertCategorical(self,train_s):
        catVar = ['PARENT1', 'MSTATUS', 'RED_CAR', 'REVOKED']
        
        # Convert categorical to dummies
        train_c = Preprocessor.cat2int(self, train_s, catVar)

        # Convert binary to dummies
        # converting SEX
        if 'SEX' in train_c.columns:
            train_c['GENDER'] = train_s['SEX'].apply(lambda x: 1 if 'M' in x else 0)
            train_c.drop(['SEX'],axis=1,inplace=True)

        # converting CAR_USE, URBANICITY
        if 'CAR_USE' in train_c.columns:
            train_c['COMMERCIAL_CAR_USE'] = train_s['CAR_USE'].apply(lambda x: 1 if 'commercial' in x.lower() else 0)
            train_c.drop(['CAR_USE'],axis=1,inplace=True)

        if 'URBANICITY' in train_c.columns:
            train_c['URBAN_CAR'] = train_s['URBANICITY'].apply(lambda x: 1 if 'urban' in x.lower() else 0)
            train_c.drop(['URBANICITY'],axis=1,inplace=True)

        # Convert EDUCATION, JOB, CAR_TYPE
        if 'EDUCATION' in train_c.columns:
            train_c['EDUCATION'] = train_c['EDUCATION'].apply(lambda x: 'Elementary Education' if '<high school' in x.lower() else x)
            # Insert dummy variables and Drop the original variable
            train_c = train_c.join(pd.get_dummies(train_c.EDUCATION.str.upper())).drop(['EDUCATION'],axis=1)
        if 'JOB' in train_c.columns:
            train_c = train_c.join(pd.get_dummies(train_c.JOB.str.upper())).drop(['JOB'],axis=1)
        if 'CAR_TYPE' in train_c.columns:
            train_c = train_c.join(pd.get_dummies(train_c.CAR_TYPE.str.upper())).drop(['CAR_TYPE'],axis=1)

        train_c.columns = train_c.columns.str.replace("Z_","")
        train_c.columns = train_c.columns.str.replace(" ","_")
        
        return train_c

