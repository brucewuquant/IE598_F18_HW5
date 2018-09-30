# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:35:09 2018

@author: wdz
"""
import copy

from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
#EDA

df_wine.describe()

def make_heatmap(df, cols):
    corMat = df[cols].corr()
    sns.heatmap(corMat, cbar = True, annot = False, square = True, fmt ='.2f',
                annot_kws = {'size' : 10}, yticklabels = cols, xticklabels = cols)
    plt.show()

make_heatmap(df_wine, df_wine.columns)

def make_boxplot(df, cols):
    
    #standardize data without any change to the original DataFrame
    tmp1 = copy.deepcopy(df)
    #vectorization way
    normalized_df=(tmp1- tmp1.mean())/tmp1.std()
    #if we instead want to use min-max: normalized_df=(df-df.min())/(df.max()-df.min())
    sns.boxplot(data =normalized_df, orient = 'h') #orient for better display
    plt.show()
    """
    alternatively using for loop
    for col in cols:
        col_data = tmp1[col]
        tmp1[col] = (col_data - col_data.mean())/ col.std()
    sns.boxplot
    plt.show()
    """

make_boxplot(df_wine, df_wine.columns)

"""
def make_scatter(df, cols):
    sns.pairplot(df[cols], size = 2.5)
    plt.show()

make_scatter(df_wine, df_wine.columns)
"""
#preprocessing
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2,
                                                    stratify = y,
                                                    random_state = 42)
sc  = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#Shape of X_train_std is (124, 13), shape of Y_train is (124,)
# np.cov np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)m: array_like
#A 1-D or 2-D array containing multiple variables and observations.
#Each row of m represents a variable, and each column a single observation of all those variables. Also see rowvar below.


#util functions

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)


def print_accuracy(clf, X_train, y_train, X_test, y_test):
    cv_score = np.mean(cross_val_score(clf, X_train, y_train, cv = 5 ,scoring ='accuracy'))
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)
    scores = [cv_score, train_score, test_score]
    print('CV score: %.5f, train_score: %.5f, test_socre: %.5f' % tuple(scores))
    return scores
    #hard coding pca



cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
#make each term sums up to 1
var_exp  =[(i/tot)for i in sorted(eigen_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha = 0.5, align = 'center', 
        label = 'individual explained variance')
plt.step(range(1,14), cum_var_exp, where = 'mid', 
         label = 'cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix w:\n', w)

X_train_std[0].dot(w)


X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()

# ## Principal component analysis in scikit-learn



pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)




plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('logistic regression train PCA')
# plt.savefig('images/05_04.png', dpi=300)
plt.show()




plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.title('logistic regression test PCA')
plt.tight_layout()
# plt.savefig('images/05_05.png', dpi=300)
plt.show()

# Logistic Regression(baseline model)


#Using one vs rest
lr_base = LogisticRegression(multi_class = 'ovr')
lr_base.fit(X_train_std, y_train)
#use lr_base.fit(X_train, y_train) for unstandardaized datasets


#SVM
svm = SVC(kernel = 'linear', C=1.0)
svm.fit(X_train_std, y_train)
#use svm.fit(X_train, y_train) for unstandardaized datasets
#PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

logreg_pca = LogisticRegression(multi_class = 'ovr')
logreg_pca.fit(X_train_pca, y_train)
svm_pca = SVC(kernel = 'linear', C = 1.0)
svm_pca.fit(X_train_pca, y_train)

#LDA
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

logreg_lda = LogisticRegression(multi_class = 'ovr')
logreg_lda.fit(X_train_lda, y_train)
svm_lda = SVC(kernel ='linear',C=1.0)
svm_lda.fit(X_train_lda, y_train)

#KPCA
kpca = KernelPCA(n_components =2 , kernel = 'rbf', gamma = 1)
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)

logreg_kpca=LogisticRegression(multi_class='ovr')
logreg_kpca.fit(X_train_kpca,y_train)
svm_kpca=SVC(kernel='linear',C=1.0)
svm_kpca.fit(X_train_kpca, y_train)

#define a dictionary to save models
DimRed_models = {
        'logreg_base': [lr_base, X_train_std, X_test_std],
        'svm':[svm, X_train_std, X_test_std],
        'logreg_pca':[logreg_pca, X_train_pca, X_test_pca], 
         'svm_pca':[svm_pca, X_train_pca, X_test_pca],
         'log_lda':[logreg_lda, X_train_lda, X_test_lda],
         'svm_lda':[svm_lda,X_train_lda, X_test_lda], 
         'log_kpca': [logreg_kpca, X_train_kpca, X_test_kpca],
         'svm_kpca': [svm_kpca, X_train_kpca, X_test_kpca]}


for i in DimRed_models:
    print ('now we are using model', i)
    print('printing accuracy scores for', i)
    scores = print_accuracy(DimRed_models[i][0],
                            DimRed_models[i][1],y_train,DimRed_models[i][2], y_test)


plt.figure()
plot_decision_regions(X_test_pca, y_test, logreg_pca, resolution=0.02)
plt.title('Logreg pca')
plt.legend(loc='lower left') 
plt.figure()
plot_decision_regions(X_test_pca, y_test, svm_pca,resolution=0.02)
plt.title('svm pca')
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_lda, y_test, logreg_lda, resolution=0.02)
plt.title('logreg lda')
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_lda, y_test, svm_lda,resolution=0.02)
plt.title('svm lda')
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_kpca, y_test, logreg_kpca, resolution=0.02)
plt.title('kpca logreg')
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_kpca, y_test, svm_kpca,resolution=0.02)
plt.title('svm kpca')
plt.legend(loc='lower left')

#selecting best gamma for kpca
gammas = [0.01, 0.077, 0.1, 1.0, 10]

test_log = []
test_svm = []

for gamma in gammas:
    kpca = KernelPCA(n_components= 2, kernel = 'rbf', gamma = gamma)
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    
    logreg_kpca = LogisticRegression(multi_class = 'ovr')
    logreg_kpca.fit(X_train_kpca, y_train)
    
    svm_kpca = SVC(kernel = 'linear', C=1.0)
    svm_kpca.fit(X_train_kpca, y_train)
    
    print('\nfor gamma ', gamma)
    print('\n the corresponding score is ')
    print_accuracy(logreg_kpca, X_train_kpca, y_train, X_test_kpca, y_test)
    print_accuracy(svm_kpca, X_train_kpca, y_train, X_test_kpca, y_test)
    test_log.append(print_accuracy(logreg_kpca, X_train_kpca, y_train, X_test_kpca, y_test)[2])
    test_svm.append(print_accuracy(svm_kpca, X_train_kpca, y_train, X_test_kpca, y_test)[2])


best_gamma_log = [i for i, scor in enumerate(test_log) if scor == max(test_log)]
best_gamma_svm =  [i for i, scor in enumerate(test_svm) if scor == max(test_svm)]

for i in range(0, len(best_gamma_log)):
    
   

    
    print( '(simutaneously) best gamma for log test score is(are) ',gammas[best_gamma_log[i]])
    
for i in range(0, len(best_gamma_svm)):
    
   

    
    print( '(simutaneously) best gamma for svm test score is(are) ',gammas[best_gamma_svm[i]])
#somehow I failed to do gridsearch on tuning kpca gamma
#requires further study
print("My name is DIZHOU WU")
print("My NetID is: dizhouw2 ")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")








