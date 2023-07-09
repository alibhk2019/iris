import joblib


def predict(data):
    clf = joblib.load("rf_model.sav", mmap_mode=None)
    return clf.predict(data)
