##########################  SENTIMENT ANALYSIS OF TWEETS  ###########################
# This is a sentiment analysis project.
# It includes the analysis of tweets that have some label based on sentiments.
# Tags include 3 classes: negative, positive, and neutral.
# It aims to predict the sentiment labels of new tweets by creating a model on tweets labeled with sentiment.

########################## Importing Library and Settings  ###########################
from warnings import filterwarnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pytz

filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


################################################
# 1. Exploratory Data Analysis
################################################
def columns_info(dataframe):
    columns, dtypes, unique, nunique, nulls = [], [], [], [], []

    for cols in dataframe.columns:
        columns.append(cols)
        dtypes.append(dataframe[cols].dtype)
        unique.append(dataframe[cols].unique())
        nunique.append(dataframe[cols].nunique())
        nulls.append(dataframe[cols].isnull().sum())

    return pd.DataFrame({"Columns": columns,
                         "Data_Type": dtypes,
                         "Unique_Values": unique,
                         "Number_of_Unique": nunique,
                         "Missing_Values": nulls})


########################## Loading  The Date  ###########################
tweets = pd.read_csv("datasets/tweets_labeled.csv", parse_dates = ["date"])
tweets_21 = pd.read_csv("datasets/tweets_21.csv", parse_dates = ["date"])
df = tweets.copy()
df_test = tweets_21.copy()
df.head()
df_test.head()


##########################  Functions  ###########################
def load_data():
    train = pd.read_csv("datasets/tweets_labeled.csv", parse_dates = ["date"])
    test = pd.read_csv("datasets/tweets_21.csv", parse_dates = ["date"])
    dataframe = train.copy()
    dataframe_test = test.copy()
    return dataframe, dataframe_test


df, df_test = load_data()
columns_info(df)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################
df.isnull().sum()
df = df.dropna().reset_index(drop = True)

df["date"] = df["date"].apply(lambda x: x.astimezone(pytz.timezone('Etc/GMT-3')))

df["season"] = np.where(df["date"].dt.month.isin([9, 10, 11]), "autumn",
                        np.where(df["date"].dt.month.isin([6, 7, 8]), "summer",
                                 np.where(df["date"].dt.month.isin([3, 4, 5]), "spring", "winter")))

df["day"] = df["date"].dt.day

df["periods"] = df["date"].dt.hour
df["periods"] = np.where((df["periods"] >= 2) & (df["periods"] < 6), "period_1",
                         np.where((df["periods"] >= 6) & (df["periods"] < 10), "period_2",
                                  np.where((df["periods"] >= 10) & (df["periods"] < 14), "period_3",
                                           np.where((df["periods"] >= 14) & (df["periods"] < 18), "period_4",
                                                    np.where((df["periods"] >= 18) & (df["periods"] < 22), "period_5",
                                                             "period_6")))))

label = {1: "positive",
         -1: "negative",
         0: "neutral"}

df["label"] = df["label"].replace(label)

df.head()
df.info()


def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


label_encoder(df, "label")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")


for col in cat_cols[1:]:
    target_summary_with_cat(df, "label", col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby([numerical_col, target]).agg({numerical_col: "count"}), end = "\n\n\n")


for col in num_cols[2:]:
    target_summary_with_num(df, "label", col)

df["tweet"] = df["tweet"].apply(lambda x: x.lower())
df = df.sample(frac=1).reset_index(drop=True)
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(df["tweet"])

y = df["label"]


##########################  Functions  ###########################
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


X_tf_idf_word, y, df = data_processing(df)


######################################################
# 3. Models
######################################################
log_model = LogisticRegression(random_state = 17).fit(X_tf_idf_word, y)

y_pred = log_model.predict(X_tf_idf_word)

print(classification_report(y_pred, y))

cross_val_score(log_model, X_tf_idf_word, y, cv = 5).mean()

df_test["tweet"] = df_test["tweet"].apply(lambda x: x.lower())
df_test.head()
df_test.info()

X_tf_idf_word_data = tf_idf_word_vectorizer.fit(df["tweet"]).transform(df_test["tweet"])
y_pred = log_model.predict(X_tf_idf_word_data)
df_test["label"] = y_pred

##########################  Functions  ###########################
def model_and_predict(X, y, train, test, random_state=17):
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


df_test = model_and_predict(X_tf_idf_word, y, df, df_test)
