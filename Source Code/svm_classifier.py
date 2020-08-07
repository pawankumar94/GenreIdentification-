# Implementing SVM Model with K-Fold Validation using Sklearn Librarires

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import metrics
import scikitplot as skplt
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_learning_curves
# Importing datasets

chunk_with_syn_X = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/x_train_syn.csv')
chunk_with_syn_y = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/y_train_syn.csv')

chunk_without_syn_X = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/chunk_data_without_syn.csv')
chunk_without_syn_Y = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/chunk_label_without_syn.csv')

book_lvl_train_x= pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/X_train_booklevel_scaleddown.csv',index_col=[0])
book_lvl_train_y = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/y_train_booklevel_scaleddown.csv',index_col=[0])

test_x = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/test_chunk_data_complete_unscalled.csv')
test_y = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/test_chunk_data_complete_unscalled_y.csv')

#unlabelled_data = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/unlabelled_dat.csv')

#unlabelled_data.pop('genre')
#unlabelled_data.pop('book_id')


#book_lvl_train_x = book_lvl_train_x.drop(['Unnamed:0'], axis=1)
# Model

def svm_model(bal,X, Y, ker,folds):

    svm_clf = svm.SVC(kernel=ker,class_weight=bal)
    scores = cross_val_score(svm_clf, X, Y,cv = folds, scoring = 'f1_weighted')#
    return (scores)

#Experiment 1: Kernel = polynomial, folds = 10, on chunk level
exp1= svm_model(X=chunk_without_syn_X,Y=chunk_without_syn_Y, folds=5, ker='poly',bal=None)
exp1_mean = np.mean(exp1)
print(exp1_mean)
print('Exp1: Kernel:Polynomial, Data:Chunks without Syn, Balanced:No = {}'.format(exp1))

#Experiment 2: .Kernel = polynomial, Balanced = Yes
exp2 = svm_model(X=chunk_without_syn_X,Y=chunk_without_syn_Y, folds=5, ker='poly',bal='balanced')
exp2_mean = np.mean(exp2)
print(exp2_mean)
print('Exp2: Kernel:polynomial, Data:Chunks without Syn, Balanced:Yes = {}'.format(exp2))

#Experiment 3: .Kernel = Polynomial
exp3 = svm_model(X=chunk_with_syn_X,Y=chunk_with_syn_y, folds=5, ker='poly',bal=None)
exp3_mean = np.mean(exp2)
print(exp3_mean)
print('Exp3: Kernel:Polynomial, Data:Chunks with Syn, Balanced:No = {}'.format(exp3))

#Experiment 4: .Kernel = Polynomial
exp4 = svm_model(X=chunk_with_syn_X,Y=chunk_with_syn_y, folds=5, ker='poly',bal='balanced')
exp4_mean = np.mean(exp4)
print(exp4_mean)
print('Exp4: Kernel:Polynomial, Data:Chunks with Syn, Balanced:Yes = {}'.format(exp4))

#Experiment 5: with book_lvl_data
exp5 = svm_model(X=book_lvl_train_x,Y=book_lvl_train_y, folds=5, ker='poly',bal=None)
exp5_mean = np.mean(exp5)
print(exp5_mean)
print('Exp5: Kernel:Polynomial, Data:Book_lvl_data, Balanced:No = {}'.format(exp5))

#Experiment 6: chunk lvl with rbf kernel
exp6 = svm_model(X=chunk_without_syn_X,Y=chunk_without_syn_Y, folds=5, ker='rbf',bal=None)
exp6_mean = np.mean(exp6)
print('Exp6: Kernel:Polynomial, Data:Chunks without Syn, Balanced:No = {}'.format(exp6))

#Plotting the F1 Scores

names = ['exp1','exp2', 'exp3', 'exp4', 'exp5']
values = [exp1_mean,exp2_mean,exp3_mean, exp4_mean,exp5_mean]

fig = plt.figure(figsize = (9,9))
ax1 = fig.add_subplot(211)
ax1.set_ylabel('Mean F1_Score')
ax1.set_xlabel('Experiments')
ax1.set_title('SVM with KFold Validation Results')
plt.bar(names,values)
plt.show()
