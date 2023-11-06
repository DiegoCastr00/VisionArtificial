def knn(df, k=5):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(k) 
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy}')
    print(report)