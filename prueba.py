import matplotlib.pyplot as plt
import numpy as np

def plano_cartesiano(coordenadas):
    
    x_coords, y_coords = zip(*coordenadas)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')

    ax.scatter(x_coords, y_coords, color='r')
    plt.grid(True)
    # Guardar la imagen en un archivo
    fig.savefig('plano_actualizado.png')
    filename = 'plano_actualizado.png'
    return filename

def activation_function(z):
    return 1 / (1 + np.exp(-z))

# Función para graficar la separación de datos
def plot_decision_boundary(X, y, w, b):
    X = np.array(X)
    y = np.array(y)
    
    # Crear una figura con un tamaño específico en pulgadas
    plt.figure(figsize = (8,8))

    # Enmallado de puntos para visualización
    xx, yy = np.meshgrid(np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, 50),
                         np.linspace(np.min(X[:,1])-1, np.max(X[:,1])+1, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([activation_function(np.dot(point, w) + b) for point in grid])

    # Colorear los puntos del enmallado
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, levels=np.linspace(np.min(Z), np.max(Z), 3))


    # Puntos de entrenamiento
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Títulos y etiquetas
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    # Mostrar gráfico
    
    filename = 'plano_actualizado.png'
    plt.savefig(filename)
    return filename