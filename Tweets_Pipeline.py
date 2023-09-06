import joblib
import helpers


def main():
    df, df_test = helpers.load_data()
    X_tf_idf_word_matrix, y, df = helpers.data_processing(df)
    df_test = helpers.model_and_predict(X_tf_idf_word_matrix, y, df, df_test)
    joblib.dump(df_test, "miuul_homework/Tweets/tweets_label_predicts.pkl")
    return df_test


if __name__ == "__main__":
    print("başla")
    try:
        main()
    except Exception as e:
        print(e)
