from sklearn.ensemble import RandomForestClassifier 
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

    runRandomForest(leaftrainx, leaftrainy, leafvalx, leafvaly, leaftestx, leaftesty)

def runRandomForest(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = RandomForestClassifier(random_state=0, max_depth=None, max_features=3, criterion='entropy')
    clf.fit(train_x, train_y)

    score = clf.score(validation_x, validation_y)
    score2 = clf.score(test_x, test_y)
    print('Results for Random Forest')
    print('validation data: ', score) # 0.7794117647058824
    print('test data: ', score2) # 0.7164179104477612
    print('\n')

if __name__ == '__main__':
    main()