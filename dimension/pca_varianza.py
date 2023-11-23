def PCA_varianza(X):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    # Estandarizar los datos)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calcular la matriz de covarianza
    cov_matrix = np.cov(X_scaled, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    # Calcular la varianza acumulada
    total_variance = np.sum(eigenvalues_sorted)
    variance_explained = np.cumsum(eigenvalues_sorted) / total_variance
    for i, explained_variance in enumerate(variance_explained):  
        print(f"Componente Principal {i+1}: {explained_variance:.4f}")

    plt.plot(range(1, len(variance_explained) + 1), variance_explained, marker='o', linestyle='--')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Acumulada Explicada')
    plt.title('Varianza Acumulada en PCA del conjunto de datos')
    plt.grid(True)
    plt.show()
    