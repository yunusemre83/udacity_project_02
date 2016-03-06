# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 08:59:42 2016

@author: yemre
"""
# Import Libraries
import numpy as np
import pandas as pd
import pylab as pl

from sklearn import metrics

from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from matplotlib.backends.backend_pdf import PdfPages
import time

from sklearn.decomposition import RandomizedPCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# FUNCTIONS
# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


def train_classifier(clf, X_train, y_train):
    print ("Training {}...".format(clf.__class__.__name__))
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print ("Done!\nTraining time (secs): {:.3f}".format(end - start))

def train_classifier_no_print(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    return (end-start)

def predict_labels(clf, features, target):
    print ("Predicting labels using {}...".format(clf.__class__.__name__))
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print ("Done!\nPrediction time (secs): {:.3f}".format(end - start))
    return metrics.f1_score(target.values, y_pred, pos_label='yes')

def predict_labels_no_print(clf, features, target):
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    return [(end-start),metrics.f1_score(target.values, y_pred, pos_label='yes')]


# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print ("------------------------------------------")
    print ("Training set size: {}".format(len(X_train)))
    train_classifier(clf, X_train, y_train)
    print ("F1 score for training set: {}".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {}".format(predict_labels(clf, X_test, y_test)))

def train_predict_no_print(clf, X_train, y_train, X_test, y_test):
    run_time = train_classifier_no_print(clf, X_train, y_train)
    return run_time, predict_labels_no_print(clf, X_train, y_train), predict_labels_no_print(clf, X_test, y_test)

def performance_score(label, prediction):
    return metrics.f1_score(label,prediction,pos_label='yes')
    
#Fine-Tuned Model or Out of the box model(Simple Model)
Fine_Tuned_Models = True
# Number of Loops in Model Averaging
N_loop_stats_collect=500;
# Training Size
training_size = [300,200,100];

# PCA parameters
PCA_on = False;
n_components = 15   # Number of Features after PCA

# Pdf Page
pp = PdfPages('results_finetunemodel_{}_pca_{}_.pdf'.format(Fine_Tuned_Models,PCA_on))        

# Read data
student_data = pd.read_csv('student-data.csv',sep=',')
print ("Student data read successfully!")
# Note: The last column 'passed' is the target/label, all other are feature columns

# Compute desired values - replace each '?' with an appropriate expression/function call
n_students = len(student_data.index)
n_features = len(student_data.columns)-1
n_passed = len(student_data[student_data['passed']=='yes'].index)
n_failed = len(student_data[student_data['passed']=='no'].index)
grad_rate = 100 * n_passed/(n_passed+n_failed)
print ("Total number of students: {}".format(n_students))
print ("Number of students who passed: {}".format(n_passed))
print ("Number of students who failed: {}".format(n_failed))
print ("Number of features: {}".format(n_features))
print ("Graduation rate of the class: {:.2f}%".format(grad_rate))


# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print ("Feature column(s):-\n{}".format(feature_cols))
print ("Target column: {}".format(target_col))

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print ("\nFeature values:-")
print (X_all.head())  # print the first 5 rows

# Pre-process student data to change class-objects to one-hot coding values as [0 1 0 0 0 0] etc.
X_all = preprocess_features(X_all)
print ("Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns)))

# Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, test_size=n_students-300)
print ("Training set: {} samples".format(X_train.shape[0]))
print ("Test set: {} samples".format(X_test.shape[0]))
# Note: If you need a validation set, extract it from within training data

if (Fine_Tuned_Models==False):
    Test_models = [tree.DecisionTreeClassifier(random_state=12),KNeighborsClassifier(weights='distance'),svm.SVC(kernel='rbf'),AdaBoostClassifier(random_state=13),RandomForestClassifier(random_state=10)];
else:
    ## For debug purposes
    #Test_models = [tree.DecisionTreeClassifier(max_depth=4,random_state=12)];
    Test_models = [tree.DecisionTreeClassifier(max_depth=4,random_state=12),KNeighborsClassifier(weights='distance'),svm.SVC(kernel='rbf',gamma=0.1),AdaBoostClassifier(learning_rate=0.5,random_state=13),RandomForestClassifier(n_estimators=14,random_state=10)];

Test_model_names = ['Decision Tree','KNN','SVM-RBF','Adaboost','Random-Forest']
# Train a model
# Choose a model, import it and instantiate an object

if (PCA_on==True):
    print("Extracting the top {} princable components from {} features".format(n_components, X_train.shape[0]))
    t0 = time.time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    t1 = time.time()
    print("PCA Training in {:0.3f}s".format(t1 - t0))
    
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time.time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    t1 = time.time()
    print("PCA Transform for training and test set in {:0.3f}s".format(t1 - t0))
    
    X_train = X_train_pca
    X_test = X_test_pca

for model_i in range(len(Test_models)):
    clf = Test_models[model_i]
    name = Test_model_names[model_i]    
    # Note: Keep the test set constant
    for fit_train_size in training_size:
        train_predict(clf, X_train[0:fit_train_size], y_train[0:fit_train_size], X_test, y_test)
        print ("Estimator {}:  {}".format(name,clf))    

#############  Average each Model to acquire confident statistics
model_stats = np.zeros([len(Test_models),len(training_size),6]);
stats_name = ['Training Time','Prediction Time of Training','Training F1','Prediction Time of Test','Test F1']
pca_stats = np.zeros([2,1])
for model_i in range(len(Test_models)):
    clf = Test_models[model_i]
    name = Test_model_names[model_i]
    for loop_i in range(N_loop_stats_collect):
        X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, test_size=n_students-300)
        if (PCA_on==True):      
            # PCA transform      
            t0 = time.time()
            pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
            t1 = time.time()
            pca_stats[0]=pca_stats[0]+(t1-t0)
            t0 = time.time()
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            t1 = time.time()
            pca_stats[1]=pca_stats[1]+(t1-t0)
            X_train = X_train_pca
            X_test = X_test_pca
        
        for ind, fit_train_size in enumerate(training_size):
            train_run_time,train_predict_stats,test_predict_stats=train_predict_no_print(clf, X_train[0:fit_train_size], y_train[0:fit_train_size], X_test, y_test)
            if (name=='Decision Tree') :
                model_stats[model_i,ind,5]=model_stats[model_i,ind,5] + clf.tree_.max_depth
            model_stats[model_i,ind,0]=model_stats[model_i,ind,0]+train_run_time
            model_stats[model_i,ind,1]=model_stats[model_i,ind,1]+train_predict_stats[0]
            model_stats[model_i,ind,2]=model_stats[model_i,ind,2]+train_predict_stats[1]
            model_stats[model_i,ind,3]=model_stats[model_i,ind,3]+test_predict_stats[0]
            model_stats[model_i,ind,4]=model_stats[model_i,ind,4]+test_predict_stats[1]
            
    
    print("Classifier: {} Averaged over {} times ".format(name,N_loop_stats_collect))
    print("       Training Time - Prediction Time of Training - Training F1 - Prediction Time of Test - Test F1 - Tree Depth")
    print("N=300  {} ".format(model_stats[model_i,0:1]/N_loop_stats_collect))
    print("N=200  {} ".format(model_stats[model_i,1:2]/N_loop_stats_collect))
    print("N=100  {} ".format(model_stats[model_i,2:3]/N_loop_stats_collect))

if (PCA_on==True):  
    print("PCA Stats: Time for PCA Training: {}s , Time for PCA Transform: {}s".format(pca_stats[0]/N_loop_stats_collect/len(Test_models),pca_stats[1]/N_loop_stats_collect/len(Test_models)))    
   
# Save data and figures into pdf    
for stat_i in range(len(stats_name)):
    pl.figure()
    index = np.arange(len(Test_models))
    bar_width = 0.15
    opacity = 0.4
    rects1 = pl.bar(index, model_stats[:,0,stat_i]/N_loop_stats_collect, bar_width,
                     alpha=opacity,
                     color='b',
                     label='300')
    
    rects2 = pl.bar(index + bar_width, model_stats[:,1,stat_i]/N_loop_stats_collect, bar_width,
                     alpha=opacity,
                     color='r',
                     label='200')
                     
    rects3 = pl.bar(index + 2*bar_width, model_stats[:,2,stat_i]/N_loop_stats_collect, bar_width,
                     alpha=opacity,
                     color='g',
                     label='100')                 
    
    pl.xlabel('Classifiers')
    pl.ylabel('Scores')
    pl.title('{}'.format(stats_name[stat_i]))
    pl.xticks(index + 3/2*bar_width, (Test_model_names[0], Test_model_names[1], Test_model_names[2], Test_model_names[3], Test_model_names[4]))
    pl.legend(bbox_to_anchor=(1.2, 1),loc=1,borderaxespad=0.)
    pl.tight_layout()
    pl.grid()
    pl.savefig(pp,format='pdf',bbox_inches='tight')
    pl.show()
    

pp.close()

##############   Grid Search
selected_model = 0
print ("Selected Model {}".format(Test_model_names[selected_model]))

parameters_DT = {'max_depth':(list(range(3,11,1)))}
parameters_KNN = {'n_neighbors':(list(range(1,10,1))),'weights':('uniform','distance')}
parameters_SVM = {'C':(0.1,1,10,100),'kernel':('linear','rbf'),'gamma':(1/(n_features-20),1/(n_features-10),1/n_features,1/(n_features+10),1/(n_features+10))}
parameters_Adaboost = {'base_estimator':(tree.DecisionTreeClassifier(max_depth=1),tree.DecisionTreeClassifier(max_depth=2)),'learning_rate':(0.3,0.5,0.75,1,2)}
parameters_RF = {'n_estimators':(list(range(6,17,2))),'max_depth':(list(range(2,11,2)))}
parameters = [parameters_DT,parameters_KNN,parameters_SVM,parameters_Adaboost,parameters_RF]

gridsearch_scorer = metrics.make_scorer(performance_score)

reg = GridSearchCV(estimator=Test_models[selected_model], param_grid=parameters[selected_model],scoring=gridsearch_scorer)

if (PCA_on==True):      
    # PCA transform 
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_all)
    X_all_pca = pca.transform(X_all)
    X_all = X_all_pca
reg.fit(X_train,y_train)
print ("Best Model: {}".format(reg.best_estimator_))



