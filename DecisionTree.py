
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *,value=None):
        self.feature = feature        
        self.threshold = threshold    
        self.left = left              
        self.right = right            
        self.value = value            
        
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None 
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root=self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        #first check stopping criteria
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        featidx=np.random.choice(n_feats,self.n_features,replace=False)
        #find the best split
        bfeature,bthreshold=self._best_split(X,y,featidx)
                
            
            
        #create child nodes
        leftidx ,rightidx=self._split(X[:,bfeature],bthreshold)
        left=self._grow_tree(X[leftidx,:],y[leftidx],depth+1)
        right=self._grow_tree(X[rightidx,:],y[rightidx],depth+1)
        return Node(bfeature,bthreshold,left,right)
            
    def _best_split(self,X,y,featidx):
        best_gain=-1
        splitidx,split_threshold=None,None
        for featidx in featidx:
            X_column=X[:,featidx]
            thresholds=np.unique(X_column)
            
            for thr in thresholds:
                gain=self._information_gain(y,X_column,thr)
                if gain>best_gain:
                    best_gain=gain
                    splitidx=featidx
                    split_threshold=thr
        return splitidx,split_threshold  
    
    def _information_gain(self,y,X_column,threshold):
        #parent Entropy
        parent_entropy=self._entropy(y)
        #children creation
        leftidx,rightidx=self._split(X_column,threshold)
        if len(leftidx)==0 or len(rightidx)==0:
            return 0
        #calc weighted avg entropy of children
        n=len(y)
        n_l,n_r=len(leftidx),len(rightidx)
        e_l,e_r=self._entropy(y[leftidx]),self._entropy(y[rightidx])
        child_entropy= (n_l/n)*e_l+(n_r/n)*e_r
        #calculate information gain   
        information_gain=parent_entropy-child_entropy
        return information_gain
    def _entropy(self,y):
        hist=np.bincount(y)
        ps=hist/len(y)
        for p in ps:
            return -np.sum([ p*np.log(p) for p in ps if p>0])
            
    def _split(self,X_column,split_thresh):
        leftidx=np.argwhere(X_column<=split_thresh).flatten()
        rightidx=np.argwhere(X_column>split_thresh).flatten()
        return leftidx,rightidx
            
    def _most_common_label(self,y):
        counter=Counter(y)
        value=counter.most_common(1)[0][0]
        return value
        
            
        
    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)
        