import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import utilities

train_houses = pd.read_csv("../../datasets/house_prices/train.csv")
train_houses = train_houses.drop(['Id'], axis=1)
train_houses = train_houses._get_numeric_data()
print("train shape: {}".format(train_houses.shape))

test_houses = pd.read_csv("../../datasets/house_prices/test.csv")
test_houses_num = test_houses.drop('Id', axis=1)
test_houses_num = test_houses_num._get_numeric_data()
print("test shape: {}".format(test_houses_num.shape))

imp = Imputer(missing_values='NaN', strategy='median', axis=1)
train_imp = imp.fit(train_houses).transform(train_houses)
test_imp = imp.fit(test_houses_num).transform(test_houses_num)

#print(train_imp[:,0:-1].shape)
#print(test_imp.shape)
model = linear_model.LinearRegression()
model.fit(train_imp[:,0:-1], train_imp[:,-1])


pred = model.predict(test_imp)
utilities.create_prediction_file(list(test_houses['Id']), list(pred), "Id", "SalePrice")