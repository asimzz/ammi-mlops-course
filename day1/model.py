from sklearn.linear_model import LogisticRegression

def fit_model(X_train, y_train):
    classifier = LogisticRegression().fit(X_train, y_train)
    return classifier

def accuracy_score(classifier, X, y):
    score = classifier.score(X, y)
    return round(score, 3) * 100