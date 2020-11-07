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
from sklearn.metrics import mean_squared_error

def main():
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

def runSGDRegressor(train_x, train_y, validation_x, validation_y, test_x, test_y):
    clf = make_pipeline(StandardScaler(), SGDRegressor(random_state=1, penalty='l2', n_iter_no_change=5, eta0=0.5, power_t=0.5))
    
    clf.fit(train_x, train_y)

    pred = clf.predict(validation_x)
    pred2 = clf.predict(test_x)

    mean = mean_squared_error(pred, validation_y, squared=False)
    mean2 = mean_squared_error(pred2, test_y, squared=False)

    print('Results for SGD Regressor')
    print('valdidation data: ', mean) # 0.7624346865676033
    print('test data: ', mean2) # 0.7152749971591288
    print('\n')

def runRidge(train_x, train_y, validation_x, validation_y, test_x, test_y):
    # scaling training data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    train_y = train_y.values.reshape(-1,1)
    train_y = scaler.fit_transform(train_y)

    # scaling validation data
    validation_x = scaler.fit_transform(validation_x)
    validation_y = validation_y.values.reshape(-1,1)
    validation_y = scaler.fit_transform(validation_y)

    # scaling test data
    test_x = scaler.fit_transform(test_x)
    test_y = test_y.values.reshape(-1,1)
    test_y = scaler.fit_transform(test_y)

    clf = Ridge(random_state=1, alpha=7, solver='sag', normalize=True)

    clf.fit(train_x, train_y)

    pred = clf.predict(validation_x)
    pred2 = clf.predict(test_x)

    mean = mean_squared_error(pred, validation_y, squared=False)
    mean2 = mean_squared_error(pred2, test_y, squared=False)
    print('Results for Ridge Regression')
    print('valdidation data: ', mean)
    print('test data: ', mean2)
    print('\n')

if __name__ == '__main__':
    main()