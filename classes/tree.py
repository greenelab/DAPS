from sklearn import tree


def classify(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
    return probas_[:, 1]


def predict(x_train, x_test, y_train,):
    clf = tree.DecisionTreeRegressor()
    clf.fit(x_train, y_train)
    return(clf.predict(x_test))
