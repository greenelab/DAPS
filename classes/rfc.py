from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor


def classify(x_train, x_test, y_train, y_test):
    max_features = 4
    if (x_train.shape[1] > max_features):
        rfc = RandomForestClassifier(n_estimators=5, max_features=max_features,
                                     random_state=123)
    else:
        rfc = RandomForestClassifier(random_state=123)

    probas_ = rfc.fit(x_train, y_train).predict_proba(x_test)
    return probas_[:, 1]


def predict(x_train, x_test, y_train,):
    clf = GradientBoostingRegressor()
    clf.fit(x_train, y_train)
    return(clf.predict(x_test))
