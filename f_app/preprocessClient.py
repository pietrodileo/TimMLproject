import pandas as pd        
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocessClient(catVars,target,fullVars,newSubject,traindf):
    catVarsTr = catVars + [target]
    ### Scale and Preprocess data
    scaler = StandardScaler()
    # copy df
    newSubject_s = newSubject.copy()
    df_s = traindf.copy()
    # remove categorical variables
    df_s.drop(catVarsTr, axis=1, inplace=True)
    newSubject_s.drop(catVars, axis=1, inplace=True)
    # training set scaling 
    df_s = pd.DataFrame(scaler.fit_transform(df_s), 
                        columns=df_s.columns)
    # scale numerical variables of the new subject
    newSubject_s = pd.DataFrame(scaler.transform(newSubject_s),
                                columns=newSubject_s.columns)
    newSubject_s = newSubject_s.join(newSubject[catVars])
    # extract actual columns
    newSVars = newSubject_s.columns
    
    # convert binary string variables in numerical
    binVar = ['PARENT1', 'MSTATUS', 'RED_CAR', 'REVOKED']
        
    # Convert categorical to dummies
    newSubject_s = cat2int(newSubject_s, binVar)

    # Convert binary to dummies
    # converting SEX
    newSubject_s['GENDER'] = newSubject_s['SEX'].apply(lambda x: 1 if 'M' in x else 0)
    newSubject_s.drop(['SEX'],axis=1,inplace=True)

    # converting CAR_USE, URBANICITY
    newSubject_s['COMMERCIAL_CAR_USE'] = newSubject_s['CAR_USE'].apply(lambda x: 1 if 'commercial' in x.lower() else 0)
    newSubject_s.drop(['CAR_USE'],axis=1,inplace=True)

    newSubject_s['URBAN_CAR'] = newSubject_s['URBANICITY'].apply(lambda x: 1 if 'urban' in x.lower() else 0)
    newSubject_s.drop(['URBANICITY'],axis=1,inplace=True)

    multicatVars = ['JOB','CAR_TYPE','EDUCATION']        

    clientdf = pd.DataFrame(np.zeros((1,len(fullVars))),columns=fullVars)
    # complete the client df 
    for var in fullVars:
        if var in newSubject_s.columns:
            clientdf[var] = newSubject_s[var][0]

    # fill dummyvars
    dummyvars = ['BACHELORS', 'ELEMENTARY_EDUCATION','MASTERS', 'PHD', 'HIGH_SCHOOL', 'CLERICAL', 'DOCTOR', 
                 'HOME_MAKER','LAWYER', 'MANAGER', 'PROFESSIONAL', 'STUDENT', 'BLUE_COLLAR','MINIVAN', 
                 'PANEL_TRUCK', 'PICKUP', 'SPORTS_CAR', 'VAN', 'SUV']
    
    # convert to lower case
    newSubject_s['JOB'] = newSubject_s['JOB'].apply(lambda x: x.upper())
    newSubject_s['CAR_TYPE'] = newSubject_s['CAR_TYPE'].apply(lambda x: x.upper())
    newSubject_s['EDUCATION'] = newSubject_s['EDUCATION'].apply(lambda x: x.upper())
    
    #fill multicategorical variables
    print(newSubject_s['JOB'][0])
    for var in dummyvars:
        condition1 = var == newSubject_s['JOB'][0]
        condition2 = var == newSubject_s['CAR_TYPE'][0]
        condition3 = var == newSubject_s['EDUCATION'][0]
        if condition1 or condition2 or condition3:
            clientdf[var][0] = 1
    
    return clientdf  

# Function that converts binary variable to int
def cat2int(df, variables):
    # c is an index that tell if a conversion was made
    c = 0
    # Create a copy of the original dataframe to avoid warnings
    newdf = df.copy()
    firstidx = df.index[0]
    for var in variables:
        if type(df[var][firstidx]) == str:
            newdf[var] = df[var].apply(lambda x: 1 if 'yes' in x.lower() else 0)
            print("Variable " + var + " was converted!")
            c = 1
        else:
            # if we re-run code it will works
            print("Variable " + var + " is not categorical")
    # Return an output  
    if c == 1:
        return newdf
    else: 
        return df
