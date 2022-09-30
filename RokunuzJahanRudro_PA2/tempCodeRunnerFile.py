LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
LR_predict = LR_model.predict(X_test)