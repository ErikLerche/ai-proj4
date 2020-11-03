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
from sklearn.svm import SVC                         
from sklearn.linear_model import SGDRegressor       
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
    # runRandomForest()
    # runLogisticRegression()
    # runSVM()
    # runSGDRegressor()
    # runRidge()
    # runPrintResults()

def runDecisionTree(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = DecisionTreeClassifier(random_state = 0)
    clf.fit(train_x, train_y)

    score = clf.score(test_x, test_y)
    print(score)
if __name__ == '__main__':
    main()