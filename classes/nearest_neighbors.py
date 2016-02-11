from sklearn import neighbors

def classify(x_train, x_test, y_train, y_test):
	clf = neighbors.KNeighborsClassifier()
	clf.fit(x_train, y_train)
	return(clf.predict(x_test))

def predict(x_train, x_test, y_train,):
	clf = neighbors.KNeighborsRegressor()
	clf.fit(x_train, y_train)
	return(clf.predict(x_test))