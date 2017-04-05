from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import utilities

houses = pd.read_csv("../../datasets/house_prices/train.csv")
houses = houses.drop(['Id'], axis=1)
numerical_houses = houses._get_numeric_data()

X = houses.drop(['SalePrice'], axis=1)
y = houses['SalePrice']

#discrete_feat = [ for feat in X]

imp = Imputer(missing_values='NaN', strategy='median', axis=1)
X_imp = imp.fit(X).transform(X)

X_best = SelectKBest(mutual_info_regression, k=25).fit_transform(X_imp, y)
print(X_best)
#X_best = pd.DataFrame(X_best)
#X_best['SalePrice'] = houses['SalePrice']
