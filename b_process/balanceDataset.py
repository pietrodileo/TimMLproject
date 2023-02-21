from imblearn.over_sampling import SMOTE
import pandas as pd

# This object applies different resampling techniques to balance an unbalanced dataset

class Balancer: 

    def __init__(self):
        print('Balancer object was created\n')

    def describeSet(self,train,target):    
        # Count the samples belonging to each class
        Balancer.class_0_count, Balancer.class_1_count = train[target].value_counts()
        # Count how many samples belong to each class
        imbSet = train.copy() 
        # extract the original values
        Balancer.class_0 = imbSet[imbSet[target] == 0]
        Balancer.class_1 = imbSet[imbSet[target] == 1]
        print('Unbalanced dataset containing:')
        print('class 0:', Balancer.class_0.shape)
        print('class 1:', Balancer.class_1.shape)
    
    def resample(self,train,target):
        # Random undersample of the majority class
        class_0_under = Balancer.class_0.sample(Balancer.class_1_count)
        Balancer.train_under = pd.concat([class_0_under, Balancer.class_1], axis=0)

        # Random oversampling of the minority class
        class_1_over = Balancer.class_1.sample(Balancer.class_0_count, replace = True)
        Balancer.train_over = pd.concat([class_1_over, Balancer.class_0], axis=0)

        # SMOTE (Synthetic Minority Oversampling Technique)
        smote = SMOTE()
        train_b = train.copy()
        y = train_b[target]
        train_b.drop(target, axis=1, inplace=True)
        Balancer.train_smote, y_sm = smote.fit_resample(train_b, y)
        Balancer.train_smote[target] = y_sm
        print('Dataset resampled and balanced')

        return Balancer.train_under,Balancer.train_over,Balancer.train_smote