

def to_X_y(
        vectorizer, df_train, df_test,
        output_name, input_name="result_full_description"
):
    X_train, X_test, feature_names, vocabulary = vectorize(
        vectorizer,
        df_train[input_name],
        df_test[input_name]
    )

    y_train = df_train[output_name]
    y_test = df_test[output_name]

    return X_train, y_train, X_test, y_test, feature_names, vocabulary


def vectorize(vectorizer, documents_train, documents_test=None):
    phrases_train = []
    for document in documents_train:
        phrases_train.extend(document.split("|"))

    vectorizer.fit(phrases_train)

    X_train = vectorizer.transform(documents_train)
    feature_names = vectorizer.get_feature_names()
    vocabulary = vectorizer.vocabulary_

    if documents_test is None:
        return X_train, feature_names, vocabulary
    else:
        X_test = vectorizer.transform(documents_test)
        return X_train, X_test, feature_names, vocabulary
