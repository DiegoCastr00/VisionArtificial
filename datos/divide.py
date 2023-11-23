import numpy as np
def divide(X, y, test_size=0.2, random_state=42):
    total_samples = len(X)
    test_samples = int(total_samples * test_size)
    np.random.seed(random_state)
    test_indices = np.random.choice(total_samples, test_samples, replace=False)
    train_indices = np.array(list(set(range(total_samples)) - set(test_indices)))
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test