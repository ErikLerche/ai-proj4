from sklearn.tree import DecisionTreeClassifier  
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

    runDecisionTree(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)

def runDecisionTree(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = DecisionTreeClassifier(random_state = 10, criterion = 'entropy', splitter='best', max_features = 3, max_depth = 12)
    clf.fit(train_x, train_y)

    score = clf.score(validation_x, validation_y)
    score2 = clf.score(test_x, test_y)
    print('Results for Decision Tree')
    print(score) 
    print(score2) 
    print('\n')

if __name__ == '__main__':
    main()
