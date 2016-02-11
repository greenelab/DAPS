from sklearn import linear_model

def predict(x_train, x_test, y_train,):
	clf = linear_model.RidgeCV()
	clf.fit(x_train, y_train)
	return(clf.predict(x_test))