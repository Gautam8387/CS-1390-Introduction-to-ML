import pdb
import numpy as np
import operator
import copy
import csv

# Binary (+1, -1) prediction.  This assumes that the data points are
# the ROWS of the matrix!!  The last column of the data matrix are the
# labels.

class DTN:
    N_THRESHOLD = 4 # don't split if node has fewer examples than this
    H_THRESHOLD = .01 # don't split if node has entropy less than this
    H_REDUCTION_THRESHOLD = .001 # don't split if it doesn't reduce H by this
    index = 0
    def __init__(self, data=None, config = None):
        self.config = config
        if config != None:
            self.N_THRESHOLD = config[0]
            self.H_THRESHOLD = config[1]
            self.H_REDUCTION_THRESHOLD = config[2]
        DTN.index += 1
        self.index = DTN.index          # unique number
        self.data = data                # store the data
        self.p = None                   # prob of positive - look at set_h
        if data is not None:
            self.n = float(data.shape[0]) # number of data points
            self.indices = range(data.shape[1]-1) # feature indices
            self.set_h()                           # compute entropy
        self.splits = {}
        # The test is data[:,fi] < th
        self.fi = None                  # feature index
        self.th = None                  # threshold
        self.lc = None                  # left child
        self.rc = None                  # right child
        self.parent = None              # parent

    # Create split on feat i at value th
    def split(self, i, th):
        self.fi = i
        self.th = th
        self.lc = DTN(self.data[self.data[:, i] < th], config = self.config)
        self.rc = DTN(self.data[self.data[:, i] >= th], config = self.config)
        self.splits[i].remove(th)
    
    # Edited Function --- start
    # def split(self, i, th):
    #     self.fi = i
    #     self.th = th
    #     left_indices = self.data[:, i] < th
    #     right_indices = self.data[:, i] >= th
        
    #     self.lc = DTN(self.data[left_indices], config=self.config)
    #     self.rc = DTN(self.data[right_indices], config=self.config)
    #     self.splits[i].remove(th)
    # Edited Function --- end

    # Evalute candidate split by weighted average entropy
    def split_eval(self, i, th):
        lc = DTN(self.data[self.data[:, i] < th], config = self.config)
        rc = DTN(self.data[self.data[:, i] >= th], config = self.config)
        pl = lc.n / self.n
        pr = 1.0 - pl
        avgH = pl * lc.h + pr * rc.h
        return avgH, lc, rc

    # Entropy of class labels in this node, assumes 1, -1
    def set_h(self):
        b = .001
        npos = np.sum(self.data[:,-1] == 1) # count labels = 1
        p = (npos + b) / (self.n + b + b)
        self.p = p
        #TODO calculate entropy self.h
        self.h = -p * np.log2(p) - (1 - p) * np.log2(1 - p) # entropy calculation

    def build_tree(self):
        if self.h < self.H_THRESHOLD or self.n <= self.N_THRESHOLD:
            return
        # Find the best split
        (i, th, (h, lc, rc)) = argmax([(i, th, self.split_eval(i, th)) \
                                                   for i in self.indices \
                                                   for th in self.get_splits(i)],
                                      lambda x: -x[2][0]) # x=(a,b,(h,c,d))
         
        if (self.h - h) < self.H_REDUCTION_THRESHOLD:
            return
        # Recurse again!
        self.fi = i
        self.th = th
        self.lc = lc
        self.rc = rc
        self.lc.parent = self
        self.rc.parent = self
        self.lc.build_tree()
        self.rc.build_tree()

    # The candidate splits between the values in the training set.
    def get_splits(self, i):
        if i not in self.splits:
            d = np.sort(self.data[:,i], axis = None)
            d1 = d[:-1]
            d2 = d[1:]
            self.splits[i] = (d1 + d2) / 2.0
        return self.splits[i]

    # Classify a data point
    def classify(self, x):
        if self.fi == None:             # leaf
            return self.p               # prob of posistive
        elif x[self.fi] < self.th:
            # return ##TODO??
            return self.lc.classify(x)  # go to left child
        else:
            # return ##TODO??
            return self.rc.classify(x)  # go to right child

    def display(self, depth=0, max_depth=3):
        if depth > max_depth:
            print(depth*'  ', 'Depth >', max_depth)
            return
        if self.fi is None:
            print(depth*'  ', '=>', "%.2f"%self.p, '[', 'n=', self.n, ']')
            return
        print(depth*'  ', 'Feat', self.fi, '<', self.th, '[', 'n=', self.n, ']')
        self.lc.display(depth+1, max_depth)
        self.rc.display(depth+1, max_depth)

def DT(X, Y, config = None):
    D = np.hstack([X, Y])               # points are rows of X
    root = DTN(D, config = config)
    root.build_tree()
    return root

def classification_error_DT(dt, X, Y):
    pred = np.array([np.apply_along_axis(dt.classify,1,X)]).T - 0.5 # predicts +,-
    return np.mean(np.sign(Y * pred) > 0.0)

# Evaluate on a train/test split
def dt_eval(trainingX, trainingY, testX, testY, max_depth=5, verbose=True, config = None):
    dt = DT(trainingX, trainingY, config = config)
    acc = classification_error_DT(dt, testX, testY)
    if verbose:
        dt.display(max_depth=max_depth)
        print('Test accuracy', acc)
    return acc    

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

######################################################################
# Nearest-Neighbor Evaluation
######################################################################

def pairwise_distances(a, b):
    pairwise_diff = a[:, None, :] - b
    distances = np.einsum('ijk,ijk->ij', pairwise_diff, pairwise_diff)

    return distances

def nn_eval(trainingX, trainingY, testX, testY, verbose=True, config=None):
    num_neighbors = config if config else 1
    pair_dist = pairwise_distances(trainingX, testX)
    near_neighbor_indices = pair_dist.argpartition(num_neighbors,
                                                   axis=0)[:num_neighbors]
    nearestY = trainingY[near_neighbor_indices]
    meanY = nearestY.mean(axis=0)
    predY = 2*(meanY > 0) - 1  # predict 1 if >0, -1 o.w.
    acc = (predY == testY).mean()
    if verbose:
        print('Test_accuracy', acc)
    return acc

######################################################################
# For auto dataset (same as in HW 3, except returns data in rows)
######################################################################

def load_auto_data(path_data='auto-mpg.tsv'):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

# Feature transformations
def std_vals(data, f):
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)

def standard(v, std):
    return [(v-std[0])/std[1]]

def raw(x):
    return [x]

def one_hot(v, entries):
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

# The class (mpg) added to the front of features (points are rows)
# Note that mpg has been made a discrete variable with value 1 or -1 representing good or bad miles per gallon
def auto_data_and_labels(auto_data, features):
    features = [('mpg', raw)] + features
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    print('avg and std', std)
    print('entries in one_hot field', entries)
    findex = 0
    # Print the meaning of the features
    for (f, phi) in features[1:]:
        if phi == standard:
            print(findex, f, 'std')
            findex += 1
        elif phi == one_hot:
            for entry in entries[f]:
                print(findex, f, entry, 'one_hot')
                findex += 1
        else:
            print(findex, f, 'raw')
            findex += 1
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    np.random.seed(0)
    np.random.shuffle(data_labels)
    return data_labels[:, 1:], data_labels[:, 0:1]

# USE FOR QUESTION 1.B AND PARAMETER TUNING
# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are standard and one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', one_hot),
            ('displacement', standard),
            ('horsepower', standard),
            ('weight', standard),
            ('acceleration', standard),
            ## Drop model_year by default
            ## ('model_year', raw),
            ('origin', one_hot)]

'''
# USE FOR QUESTION 1.C and 3.B
# A small feature set
features = [('weight', standard),
            ('origin', one_hot)]
'''

'''
# USE FOR QUESTION 1.D and 3.B
# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are standard and one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('weight', raw),
            ('origin', raw)]
'''

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = load_auto_data('auto-mpg.tsv')

# Construct the standard data and label arrays
auto_data, auto_labels = auto_data_and_labels(auto_data_all, features)

######################################################################
# Apply the decision tree to the auto data
######################################################################

# Run a single train/test split
def auto_test(data, labels, pct=0.25, decision_tree=True):  # pct is for test
    model_eval = dt_eval if decision_tree else nn_eval
    X = data
    Y = labels
    (n, d) = X.shape
    indices = np.random.permutation(n)  # randomize the data set
    tx = int((1-pct)*n)                 # size of training split
    training_idx, test_idx = indices[:tx], indices[tx:]
    trainingX, testX = X[training_idx,:], X[test_idx,:]
    trainingY, testY = Y[training_idx,:], Y[test_idx,:]
    return model_eval(trainingX, trainingY, testX, testY)

# Cross validate with k folds
def auto_xval(data, labels, k=10, decision_tree=True, verbose=True, config=None):
    model_eval = dt_eval if decision_tree else nn_eval
    indices = np.random.permutation(auto_data.shape[0])
    X = data[indices,:]
    Y = labels[indices,:]
    ##TODO SPLIT DATA INTO K FOLDS   --- BEGIN
    n = X.shape[0]
    fold_size = n // k
    ##TODO SPLIT DATA INTO K FOLDS    --- END
    score_sum = 0
    for i in range(k):
        ##TODO USE ith fold for testing, remaiing for traininng    --- BEGIN
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        data_train = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        labels_train = np.concatenate((Y[:test_start], Y[test_end:]), axis=0)
        data_test = X[test_start:test_end]
        labels_test = Y[test_start:test_end]
        ##TODO USE ith fold for testing, remaiing for traininng    --- END

        if verbose == "first_tree":
            if i == 0:
                score_sum += model_eval(data_train, labels_train, data_test, labels_test, verbose = True, config = config)
            else:
                score_sum += model_eval(data_train, labels_train, data_test, labels_test, verbose = False, config = config)
        else:
            score_sum += model_eval(data_train, labels_train, data_test, labels_test, verbose = verbose, config = config)
    print('Xval accuracy', score_sum/k)
    return score_sum/k

print('Loaded non_parametric.py')

'''
# Questions 1 and 2
print(auto_xval(auto_data, auto_labels))
'''

######################################################################
# Apply the nearest neighbor model to the auto data
######################################################################

'''
# Question 3
print(auto_xval(auto_data, auto_labels, decision_tree=False))
'''