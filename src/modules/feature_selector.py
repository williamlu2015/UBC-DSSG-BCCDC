def select(selection, X_train, y_train, X_test, feature_names):
    selection.fit(X_train, y_train)
    support = selection.get_support(indices=True)

    X_train_new = X_train[:, support]
    X_test_new = X_test[:, support]
    feature_names_new = [feature_names[index] for index in support]

    return X_train_new, X_test_new, feature_names_new
