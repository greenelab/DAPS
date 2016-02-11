from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor


def classify(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=123)
    probas_ = rfc.fit(x_train, y_train).predict_proba(x_test)
    return probas_[:, 1]


def predict(x_train, x_test, y_train,):
    clf = GradientBoostingRegressor()
    clf.fit(x_train, y_train)
    return(clf.predict(x_test))
