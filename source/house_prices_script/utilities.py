from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_prediction_file(Id, y, label_id, label_y, filename='predictions.csv'):
    assert len(Id) == len(y), ValueError("Id and y have different lengths")
    with open(filename, "w") as f:
        f.write("{},{}\n".format(label_id, label_y))
        for Id, prediction in zip(Id, y):
            f.write("{},{}\n".format(Id, prediction))

def sub_na(df, method='median'):
    for col in df:
        if method=='median':
            value = df[col].median(skipna=True)
        df[col] = df[col].fillna(value)
    return df

def linRegr(model, iterations, df, test_size):
    tot_score = 0.
    tot_rmse = 0.
    for i in range(iterations):
        train, test = train_test_split(df, test_size=test_size)

        X_train = train.drop(['SalePrice'], axis=1)
        y_train = train['SalePrice']
        X_test = test.drop(['SalePrice'], axis=1)
        y_test = test['SalePrice']

        # Train the model using the training sets
        model.fit(X_train, y_train)

        # Coefficients: this is the m term in the formula f(x) = mx + q
        # print('Coefficients: \n', regr.coef_, regr.intercept_)
        pred = model.predict(X_test)
        # Mean squared error
        rmse = mean_squared_error(y_test, pred) ** 0.5
        #print("Root mean squared error: %.2f" % rmse)
        # Explained variance score: 1 is perfect prediction
        score = model.score(X_test, y_test)
        #print('Variance score: %.2f' % score)
        tot_rmse += rmse
        tot_score += score

    print('\nAverage score: %.2f' % float(tot_score / iterations))
    print('Average rmse: %.2f' % float(tot_rmse / iterations))