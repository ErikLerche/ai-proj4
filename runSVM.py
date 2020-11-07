from sklearn import svm  
from sklearn.metrics import accuracy_score
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

    runSVM(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)

def runSVM(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = svm.SVC(C = 1000.0, kernel = 'rbf', gamma = 'auto', degree = 3)
    clf.fit(train_x, train_y)

    score = clf.score(validation_x, validation_y)
    score2 = clf.score(test_x, test_y)
    print('Results for SVM')
    print(score) 
    print(score2) 
    print('\n')

if __name__ == '__main__':
    main()
