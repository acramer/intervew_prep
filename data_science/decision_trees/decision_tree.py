import pandas as pd
import numpy as np
from collections import Counter

DATASET = '../../data/banknote_auth.csv' 
def get_data():
    return pd.read_csv(DATASET,header=None)

#==================================================

# Gini Index: Cost function used by DT Algo
def calc_gini(groups):
    groups = [g[:,-1] for g in groups if g.shape[0]]
    total_size = sum(map(len,groups))
    gini = 0
    for g in groups:
        group_size = len(g)
        squared_proportions = [(c/group_size)**2 for c in Counter(g).values()] 
        gini += (1-sum(squared_proportions))*group_size/total_size
    return gini


class DecisionTree:
    class Node:
        def __init__(self,fidx=0,threshold=0,children=None,parent=None,pred=None):
            self.fidx = fidx
            self.threshold = threshold
            self.children = children if children else []
            self.parent = parent
            self.pred = pred

        def __call__(self,x):
            if self.pred is not None:
                return self.pred
            return self.children[int(x[self.fidx] >= self.threshold)](x)

    def __init__(self, data, max_depth=0):
        num_feat = data.shape[1]-1

        self.root = DecisionTree.Node()
        cur_node = self.root

        cdata = data
        sdata = []
        while cur_node is not None:
            if len(cur_node.children) == 2:
                cur_node = cur_node.parent
                cdata = sdata.pop() if sdata else None

            elif len(cur_node.children) == 1:
                sdata.append(cdata)
                cdata = cdata[cdata[:,cur_node.fidx]>=cur_node.threshold]

                new_node = DecisionTree.Node(parent=cur_node)
                cur_node.children.append(new_node)
                cur_node = new_node
            else:
                min_args = (0,cdata[0,0])
                min_gini = 1
                break_fidx_loop = False
                for fidx in range(num_feat):
                    if break_fidx_loop: break
                    for thresh in cdata[:,fidx]:
                        groups = [cdata[cdata[:,fidx]<thresh],cdata[cdata[:,fidx]>=thresh]]
                        gini = calc_gini(groups)
                        if gini < min_gini:
                            if gini > 0:
                                min_args = (fidx,thresh)
                                min_gini = gini
                            else:
                                if len(groups[0]) == 0 or len(groups[1]) == 0:
                                    cur_node.pred = cdata[0,-1]
                                    cur_node = cur_node.parent
                                    cdata = sdata.pop() if sdata else None
                                else:
                                    cur_node.children = [
                                        DecisionTree.Node(parent=cur_node,pred=groups[0][0,-1]),
                                        DecisionTree.Node(parent=cur_node,pred=groups[1][0,-1])]
                                    cur_node.threshold = thresh
                                break_fidx_loop = True
                                break

                if not break_fidx_loop:
                    cur_node.fidx = min_args[0]
                    cur_node.threshold = min_args[1]

                    if max_depth > 0 and len(sdata) >= max_depth:
                        fidx,thresh = min_args
                        groups = [cdata[cdata[:,fidx]<thresh],cdata[cdata[:,fidx]>=thresh]]
                        p0 = sorted(Counter(groups[0][:,-1]).items(),key=lambda x:(x[1],x[0]),reverse=True)[0][0]
                        p1 = sorted(Counter(groups[1][:,-1]).items(),key=lambda x:(x[1],x[0]),reverse=True)[0][0]
                        cur_node.children = [
                            DecisionTree.Node(parent=cur_node,pred=p0),
                            DecisionTree.Node(parent=cur_node,pred=p1)]

                    else:
                        sdata.append(cdata)
                        cdata = cdata[cdata[:,cur_node.fidx]<cur_node.threshold]

                        new_node = DecisionTree.Node(parent=cur_node)
                        cur_node.children.append(new_node)
                        cur_node = new_node

        
    def __call__(self,x):
        return self.root(x)

#==================================================

def main():
    df = get_data()

    dataset = df.values

    ridx = np.random.permutation(np.arange(len(dataset)))
    test_size = int(len(ridx) * 0.1)

    train_data = dataset[ridx[:test_size]]
    test_data = dataset[ridx[test_size:]]
    model = DecisionTree(train_data)

    test_labels = test_data[:,-1]
    test_data = test_data[:,:-1]
    correct = 0
    for x,y in zip(test_data,test_labels):
        if int(model(x))==int(y):
            correct+=1
    print(correct/len(test_data),'%')


if __name__ == '__main__': main()
