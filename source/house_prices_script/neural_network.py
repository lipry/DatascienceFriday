from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import utilities

houses = pd.read_csv("../../datasets/house_prices/train.csv")
houses = houses.drop(['Id'], axis=1)
numerical_houses = houses._get_numeric_data()
med_na_houses = utilities.sub_na(numerical_houses)

iterations = 50
score = 0
for i in range(iterations):
    train, test = train_test_split(med_na_houses, test_size=0.3)
    X_train = train.drop(['SalePrice'], axis=1)
    y_train = train['SalePrice']
    X_test = test.drop(['SalePrice'], axis=1)
    y_test = test['SalePrice']


    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    clf.predict(X_test)

    score += clf.score(X_test, y_test)

print(score/iterations)