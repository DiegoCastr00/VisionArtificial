import numpy as np
import matplotlib.pyplot as plt

def multiplicar_por_transpuesta(A):
    m, n = A.shape
    resultado = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            resultado[i, j] = np.dot(A[i, :], A[j, :])
    return resultado

class MDC():
    def __init__(self):
        self.classes = []
        self.means = []
    
    
    def getCentroids(self, data, target):
        data = np.array(data)
        target = np.array(target)
        valuesUnic = np.unique(target)
        self.classes = valuesUnic
        dataWtarget = np.column_stack((data, target))
        means = np.zeros((len(valuesUnic), data.shape[1]))
        for idx, value in enumerate(valuesUnic):
            filtered_data = dataWtarget[dataWtarget[:, -1] == value][:, :-1]
            means[idx, :] = np.mean(filtered_data, axis=0)
        self.means = means
    
    def predict(self, prof):
        distances = np.sqrt(np.sum((prof[:, np.newaxis, :] - self.means) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)
        predictions = self.classes[labels]
        return predictions
    
    def getRectas(self):
        recValues = []
        #x, y , m
        recValues = np.copy(self.means)
        recValues = np.hstack((recValues, np.zeros((recValues.shape[0], 1))))
        for i in range(0, self.means.shape[0]):
            prom = self.means[i, :].reshape(1, -1)
            recValues[i , -1] = - 1/2 * multiplicar_por_transpuesta(prom)[0,0]
        recFinal = recValues[0, :] - recValues[1, :]
        return recFinal
    
    def getGraphics(self, data):
        data = np.array(data)
        predictions = self.predict(data)
        recta = self.getRectas()
        
        A = recta[0]
        B = recta[1]
        C = recta[2]

        def recta(x):
            return (-A * x - C) / B
        
        A_formatted = f'{A:.2f}'
        B_formatted = f'{B:.2f}'
        C_formatted = f'{C:.2f}'

        x_vals = np.linspace(-10, 10, 100)
        y_vals = recta(x_vals)
        plt.plot(x_vals, y_vals, label=f'{A_formatted}x + {B_formatted}y - {C_formatted} = 0')

        for idx, class_label in enumerate(self.classes):
            class_data = data[predictions == class_label]
            plt.scatter(class_data[:, 0], class_data[:, 1], label=f'Class {class_label}', marker=f'${idx}$', s=100)
            
        plt.scatter(self.means[:, 0], self.means[:, 1], marker='X', s=100, color='black', label='Class Means')

        plt.xlim(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1)
        plt.ylim(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.title('MDC Predictions, Class Means, and Recta')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
