from sklearn.linear_model import LogisticRegression 
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

    outputHighestLeafAccuracy(leaftrainx, leaftrainy, leaftestx, leaftesty)

def outputHighestLeafAccuracy(train_x, train_y, test_x, test_y):
    clf = LogisticRegression(random_state = 0, C= 10000.0, solver = 'newton-cg', multi_class = 'ovr', penalty = 'l2')
    clf.fit(train_x, train_y)

    prediction = clf.predict(test_x)
    base = test_y.tolist()
    df = pd.DataFrame(base, prediction)
    df.to_excel('output.xlsx')

if __name__ == '__main__':
    main()