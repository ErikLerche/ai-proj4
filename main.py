# Models:
# Linear regression: 
# 	https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
#         https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html?highlight=ridge%20regression#sklearn.linear_model.ridge_regression
# Logistic regression: 
# 	https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
# 	https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# Decision tree:
# 	https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# 	https://scikit-learn.org/stable/modules/tree.html
# Random forest:
# 	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Support Vector Machine:
# 	https://scikit-learn.org/stable/modules/svm.html

# Evaluation Metrics:
# https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
# https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
# https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error

from sklearn.tree import DecisionTreeClassifier     
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm                        
from sklearn.linear_model import SGDRegressor   
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score    
import pandas as pd  

def main():
    leaftrain = pd.read_csv('leaf/train_data.csv')
    leaftrainx = leaftrain.iloc[:, 1:]
    leaftrainy = leaftrain['class']

    leafval = pd.read_csv('leaf/validation_data.csv') 
    leafvalx = leafval.iloc[:, 1:]
    leafvaly = leafval['class']

    leaftest = pd.read_csv('leaf/test_data.csv')
    leaftestx = leaftest.iloc[:, 1:]
    leaftesty = leaftest['class']

    runDecisionTree(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)
    runRandomForest(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)
    runLogisticRegression(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)
    runSVM(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)

    mpgtrain = pd.read_csv('auto_mpg/train_data.csv')
    mpgtrainx = mpgtrain.iloc[:, 1:]
    mpgtrainy = mpgtrain['mpg']

    mpgval = pd.read_csv('auto_mpg/validation_data.csv')
    mpgvalx = mpgval.iloc[:, 1:]
    mpgvaly = mpgval['mpg']

    mpgtest = pd.read_csv('auto_mpg/test_data.csv')
    mpgtestx = mpgtest.iloc[:, 1:]
    mpgtesty = mpgtest['mpg']

    runSGDRegressor(mpgtrainx, mpgtrainy, mpgvalx, mpgvaly, mpgtestx, mpgtesty)
    runRidge(mpgtrainx, mpgtrainy, mpgvalx, mpgvaly, mpgtestx, mpgtesty)
    # runPrintResults()

def runDecisionTree(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = DecisionTreeClassifier(random_state = 0)
    clf.fit(train_x, train_y)

    score = clf.score(test_x, test_y)
    print('Results for Decision Tree')
    print(score) # 0.5373134328358209
    print('\n')

def runRandomForest(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = RandomForestClassifier(random_state=0, max_depth=10, max_features=8, criterion='entropy')
    clf.fit(train_x, train_y)

    score = clf.score(validation_x, validation_y)
    score2 = clf.score(test_x, test_y)
    print('Results for Random Forest')
    print('validation data: ', score) # 0.7794117647058824
    print('test data: ', score2) # 0.7164179104477612
    print('\n')

def runLogisticRegression(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = LogisticRegression(random_state = 0)
    clf.fit(train_x, train_y)

    score = clf.score(test_x, test_y)
    print('Results for Logistic Regression')
    print(score) # 0.373134328358209
    print('\n')

def runSVM(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = svm.SVC()
    clf.fit(train_x, train_y)

    score = clf.score(test_x, test_y)
    print('Results for SVM')
    print(score) # 0.208955223880597
    print('\n')

def runSGDRegressor(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = make_pipeline(StandardScaler(), SGDRegressor(random_state=1, penalty='l2', n_iter_no_change=5, eta0=0.5, power_t=0.5))
    scaler = StandardScaler()
    clf.fit(scaler.transform(train_x), scaler.transform(train_y))

    score = clf.score(validation_x, validation_y)
    score2 = clf.score(test_x, test_y)
    print('Results for SGD Regressor')
    print('valdidation data: ', score) # 0.7624346865676033
    print('test data: ', score2) # 0.7152749971591288
    print('\n')

def runRidge(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = Ridge(random_state=1, solver='cholesky', max_iter=100, normalize=True)

    clf.fit(train_x, train_y)

    score = clf.score(validation_x, validation_y)
    score2 = clf.score(test_x, test_y)
    print('Results for Ridge Regression')
    print('valdidation data: ', score)
    print('test data: ', score2)
    print('\n')

if __name__ == '__main__':
    main()