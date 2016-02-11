from sklearn.svm import SVC, SVR


def classify(x_train, x_test, y_train, y_test):
    clf = SVC(probability=True, random_state=123)
    probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
    return probas_[:, 1]


def predict(x_train, x_test, y_train,):
    clf = SVR()
    clf.fit(x_train, y_train)
    return(clf.predict(x_test))
