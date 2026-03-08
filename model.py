"""
This doc contains all the functions defined in spam.ipynb.
This is just to make reusable code easier to run.

"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def load_models():
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model1_nb.pkl', 'rb') as f:
        model1 = pickle.load(f)
    with open('model2_lr.pkl', 'rb') as f:
        model2 = pickle.load(f)
    with open('model3_svm.pkl', 'rb') as f:
        model3 = pickle.load(f)
    return tfidf, model1, model2, model3


def load_data():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    data = data.drop_duplicates(subset='message')
    X = data['message']
    Y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    return x_train, x_test, y_train, y_test


def predict(model, tfidf, message):
    msg_v = tfidf.transform([message])
    pred = model.predict(msg_v)[0]
    label = "Spam" if pred == 1 else "Ham"
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(msg_v)[0]
    return label, prob


def get_confusion_matrix(model, x_test_v, y_test):
    y_pred = model.predict(x_test_v)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(model, x_test_v, y_test):
    y_pred = model.predict(x_test_v)
    report = classification_report(
        y_test, y_pred,
        target_names=['Ham', 'Spam'],
        output_dict=True
    )
    return pd.DataFrame(report).transpose().round(2)


def get_top_words(tfidf, model, n=20):
    feature_names = tfidf.get_feature_names_out()
    spam_weights = model.feature_log_prob_[1]
    ham_weights = model.feature_log_prob_[0]
    diff_weights = spam_weights - ham_weights

    top_spam_idx = np.argsort(diff_weights)[-n:][::-1]
    top_ham_idx = np.argsort(diff_weights)[:n]

    return {
        "feature_names": feature_names,
        "diff_weights": diff_weights,
        "top_spam_idx": top_spam_idx,
        "top_ham_idx": top_ham_idx
    }