import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pytz

def load_data():
    train = pd.read_csv("datasets/tweets_labeled.csv", parse_dates = ["date"])
    test = pd.read_csv("datasets/tweets_21.csv", parse_dates = ["date"])
    dataframe = train.copy()
    dataframe_test = test.copy()
    return dataframe, dataframe_test

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def data_processing(dataframe):
    temp = dataframe.copy()
    temp = temp.dropna().reset_index(drop = True)

    temp["date"] = temp["date"].apply(lambda x: x.astimezone(pytz.timezone('Etc/GMT-3')))

    temp["season"] = np.where(temp["date"].dt.month.isin([9, 10, 11]), "autumn",
                              np.where(temp["date"].dt.month.isin([6, 7, 8]), "summer",
                                       np.where(temp["date"].dt.month.isin([3, 4, 5]), "spring", "winter")))

    temp["day"] = temp["date"].dt.day

    temp["periods"] = temp["date"].dt.hour
    temp["periods"] = np.where((temp["periods"] >= 2) & (temp["periods"] < 6), "period_1",
                               np.where((temp["periods"] >= 6) & (temp["periods"] < 10), "period_2",
                                        np.where((temp["periods"] >= 10) & (temp["periods"] < 14), "period_3",
                                                 np.where((temp["periods"] >= 14) & (temp["periods"] < 18), "period_4",
                                                          np.where((temp["periods"] >= 18) & (temp["periods"] < 22),
                                                                   "period_5",
                                                                   "period_6")))))

    label = {1: "positive",
             -1: "negative",
             0: "neutral"}

    temp["label"] = temp["label"].replace(label)
    label_encoder(temp, "label")

    temp["tweet"] = temp["tweet"].apply(lambda x: x.lower())
    temp = temp.sample(frac = 1).reset_index(drop = True)

    tfidf_word_vectorizer = TfidfVectorizer()
    X_tf_idf_word_matrix= tfidf_word_vectorizer.fit_transform(temp["tweet"])

    target = temp["label"]
    return X_tf_idf_word_matrix, target, temp

def model_and_predict(X, y, train, test, random_state = 17):
    temp = train.copy()
    temp_2 = test.copy()

    log_model = LogisticRegression(random_state = random_state).fit(X, y)

    temp["tweet"] = temp["tweet"].apply(lambda x: x.lower())
    temp_2["tweet"] = temp_2["tweet"].apply(lambda x: x.lower())

    tf_idf_word_vectorizer = TfidfVectorizer()
    X_tf_idf_word_data = tf_idf_word_vectorizer.fit(temp["tweet"]).transform(temp_2["tweet"])
    y_pred = log_model.predict(X_tf_idf_word_data)

    temp_2["label"] = y_pred
    return temp_2