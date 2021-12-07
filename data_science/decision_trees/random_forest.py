import pandas as pd
import numpy as np
from collections import Counter

from decision_tree import DecisionTree

DATASET = '../../data/sonar.csv' 
def get_data():
    return pd.read_csv(DATASET,header=None)

#==================================================

class RandomForest:
    def __init__(self,data,num_trees=5,num_feat=0.7,num_boot=0.7):
        total_feat = data.shape[1]-1
        total_boot = data.shape[0]
        if num_feat < 1: 
            num_feat = int(total_feat*num_feat)
        if num_boot < 1: 
            num_boot = int(total_boot*num_boot)
        assert num_feat > 0 and num_feat <  data.shape[1]
        assert num_boot > 0 and num_boot <= data.shape[0]
        self.trees = []
        self.feat_idx = []
        for _ in range(num_trees):
            fidx = list(np.random.permutation(np.arange(total_feat))[:num_feat])+[total_feat]
            eidx = np.random.permutation(np.arange(total_boot))[:num_boot]
            self.feat_idx.append(fidx)
            self.trees.append(DecisionTree(data[eidx,:][:,fidx]))

    def __call__(self,x):
        return sorted(Counter([tree(x[fidx[:-1]]) for tree,fidx in zip(self.trees,self.feat_idx)]).items(),reverse=True,key=lambda x:(x[1],x[0]))[0][0]


#==================================================

def main():
    a = list(range(10))
    
    exit()

    df = get_data()

    dataset = df.values

    ridx = np.random.permutation(np.arange(len(dataset)))
    test_size = int(len(ridx) * 0.1)

    folds = 5
    for fold in range(folds):
        ridx[]

    train_data = dataset[ridx[:test_size]]
    test_data = dataset[ridx[test_size:]]
    test_labels = test_data[:,-1]
    test_data = test_data[:,:-1]

    model = DecisionTree(train_data)
    correct = 0
    for x,y in zip(test_data,test_labels):
        if int(model(x))==int(y):
            correct+=1
    print(correct/len(test_data),'%')

    model = RandomForest(train_data,num_trees=10,num_feat=0.9)
    correct = 0
    for x,y in zip(test_data,test_labels):
        if int(model(x))==int(y):
            correct+=1
    print(correct/len(test_data),'%')


if __name__ == '__main__': main()
