"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        # Grabs the shape of X (the dimensions)
        n,d = X.shape

        # This makes prediction into an array, then makes it into a n sized array
        # with prediction for all n elements
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set

        # This will create a Counter, and put in the number
        # of times that each key appears
        c = Counter(y)

        # This will get the total value of all the items (the values combined)
        total_value = float(sum(c.values()))

        # This creates a dictionary and changes the values to probabilities
        d = dict(c)
        for key in d:
        	d[key] = (d[key] / total_value)

        self.probabilities_ = d
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)

        # Grab shape of X
        n, d = X.shape

        # Parameters:
        # self.probabilities_.keys() returns a dict_keys object
        # np.random.choice accepts a 1-D array like or int (not dict_keys)
        # for the "a" and "p" arguments
        # n is the size of the array that we want
        # p=self.probabilities_.values() grabs the probabilities of the keys

        y = np.random.choice(a=list(self.probabilities_.keys()), size=n, p=list(self.probabilities_.values()))
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    
    # Do ntrials
    for trial in range(ntrials):
    	# This splits the training and test data
    	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=trial)

    	# Fit will "create a model" that allows us to predict data
    	clf.fit(X_train, y_train)
    	y_pred_train = clf.predict(X_train)
    	y_pred_test = clf.predict(X_test)
    	train_error += 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
    	test_error += 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)

    # Calculate the errors
    train_error = train_error / float(ntrials)
    test_error = test_error / float(ntrials)

    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    print('Loading Titanic dataset...')
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    #========================================
    # part a: plot histograms of each feature
    # print 'Plotting...'
    # for i in xrange(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data

    # Accuracy score compares the two arrays, and checks whether or not it is equal to the
    # true value. It will then return a fraction based on the number of correct values
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    rand_clf = RandomClassifier()
    rand_clf.fit(X, y)
    rand_pred = rand_clf.predict(X)
    print('Finished predicting!')

    rand_error = 1 - metrics.accuracy_score(y, rand_pred, normalize=True)
    print('\t-- training error: %.3f' % rand_error)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    
    dt_clf = DecisionTreeClassifier(criterion="entropy")
    dt_clf.fit(X, y)
    dt_pred = dt_clf.predict(X)

    dt_error = 1 - metrics.accuracy_score(y, dt_pred, normalize=True)
    print('\t-- training error: %.3f' % dt_error)

    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    
    maj_train_error, maj_test_error = error(clf, X, y)
    rand_train_error, rand_test_error = error(rand_clf, X, y)
    dt_train_error, dt_test_error = error(dt_clf, X, y)

    print('Majority -- training error: %.3f' % maj_train_error)
    print('Majority -- test error: %.3f' % maj_test_error)

    print('Random -- training error: %.3f' % rand_train_error)
    print('Random -- test error: %.3f' % rand_test_error)

    print('Decision Tree -- training error: %.3f' % dt_train_error)
    print('Decision Tree -- test error: %.3f' % dt_test_error)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print('Investigating depths...')
    
    depths = np.arange(1, 21)
    training_errors = []
    testing_errors = []

    for depth in depths:
        depth_clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        train_err, test_err = error(depth_clf, X, y)
        training_errors.append(train_err)
        testing_errors.append(test_err)

   	# Plot the training errors
    print('Plotting average training errrors...')
    plt.plot(depths, training_errors, label="Decision Tree")
    plt.axhline(y=maj_train_error, color='r', label="Majority Vote")
    plt.axhline(y=rand_train_error, color='g', label="Random")
    plt.xlabel('Depth Limit')
    plt.ylabel('Average Training Error')

    # Add the labels, and the legend will display them for you
    plt.legend()
    plt.show()

   	# Plot the testing errors
    print('Plotting average testing errors...')
    plt.plot(depths, testing_errors, label="Decision Tree")
    plt.axhline(y=maj_test_error, color='r', label="Majority Vote")
    plt.axhline(y=rand_test_error, color='g', label="Random")
    plt.xlabel('Depth Limit')
    plt.ylabel('Average Testing Error')
    plt.legend()
    plt.show()

    # Convert to numpy array
    np_testing_error = np.asarray(testing_errors)
    lowest_error = np.amin(np_testing_error)
    best_depth = depths[np_testing_error.argmin()]
    print('Best depth at: ' + str(best_depth) + ' with lowest error: ' + str(lowest_error))

    ### ========== TODO : END ========== ### 
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print('Investigating training set sizes...')
    
    training_splits = np.arange(0.05, 1, 0.05)
    split_train = []
    split_test = []
    for split in training_splits:
        split_clf = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth)
        train_err, test_err = error(split_clf, X, y, test_size=split)
        split_train.append(train_err)
        split_test.append(test_err)

    # Plot the testing errors
    print('Plotting average training errrors...')
    plt.plot(training_splits, split_train, label="Decision Tree")
    plt.axhline(y=maj_test_error, color='r', label="Majority Vote")
    plt.axhline(y=rand_test_error, color='g', label="Random")
    plt.xlabel('Amount of Training Data')
    plt.ylabel('Average Testing Error')
    plt.legend()
    plt.show()

    ### ========== TODO : END ========== ###

if __name__ == "__main__":
    main()
