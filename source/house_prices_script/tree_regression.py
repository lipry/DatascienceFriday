from sklearn import tree
import pandas as pd
import utilities

houses = pd.read_csv("../../datasets/house_prices/train.csv")
houses = houses.drop(['Id'], axis=1)
numerical_houses = houses._get_numeric_data()

zero_na_houses = numerical_houses.fillna(0)

iterations = 100
model = tree.DecisionTreeRegressor()
# Create linear regression object
print("Tree Regression with NA = 0")
utilities.linRegr(model, iterations, df=zero_na_houses, test_size=0.3)
print()

med_na_houses = utilities.sub_na(numerical_houses)
print("Tree Regression with NA = median")
utilities.linRegr(model, iterations, df=med_na_houses, test_size=0.3)
print()

print("Tree Regression with NA columns removed")
drop_na_houses = numerical_houses.dropna(axis=1)
utilities.linRegr(model, iterations, df=drop_na_houses, test_size=0.3)
print()
