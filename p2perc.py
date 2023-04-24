from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')
class Perceptron(object):

    # Konstruktor, podajemy współczynik uczenia sie oraz ilość epok
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+ X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#utworzenie klasy ktora przyjmuje 2 perceptrony
class Classifier:
  #konstruktor
    def __init__(self, ppn1, ppn2):
        self.ppn1 = ppn1
        self.ppn2 = ppn2
        
    # metoda przewidywania wyniku dla 3 klas
    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 2, 1))
        def main():
    #pobranie danych iris
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]] # wybór kolumn 2 i 3 jako zmiennych
    y = iris.target # wybór zmiennej objaśnianej

    #podzial zbioru na testowy  i treningowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) # Divide downloaded data into testing and training data

    #uworzenie zbiorow do znalezienia klas
    y_train_01_subset = y_train.copy() # do znalezienia klasy 0 i inne jako -1
    y_train_02_subset = y_train.copy() # do znalezienia klasy 2 i 1 jako -1
    
    #zbior do trenowania
    X_train_01_subset = X_train.copy()

    #przypisanie wartosci tak aby rozpoznawalo tylko klase 0 
    y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_subset[(y_train_01_subset == 0)] = 1

    #przypisanie wartosci tak aby znajowalo tylko klase 2
    y_train_02_subset[(y_train == 1) | (y_train == 0)] = -1
    y_train_02_subset[(y_train_02_subset == 2)] = 1

    #wyswietlenie
    print('do znajdywania klasy 0: ', y_train_01_subset)
    print('do znajdywania klasy 2: ', y_train_02_subset)

    #tworzenie perceptronow
    ppn1 = Perceptron(eta=0.1, n_iter=300) # Perceptron 1
    #trenowanie 1
    ppn1.fit(X_train_01_subset, y_train_01_subset)

    ppn2 = Perceptron(eta=0.1, n_iter=300) # Perceptron 2
    #trenowanie 2
    ppn2.fit(X_train_01_subset, y_train_02_subset)

    #predykcja
    y1_predict = ppn1.predict(X_train)
    y2_predict = ppn2.predict(X_train)

    print('wyniki testu 1: ', y1_predict)
    print('wyniki testu 2: ', y2_predict)

    #obliczenie dokladnosci
    accuracy_1 = accuracy(ppn1.predict(X_train), y_train_01_subset)
    accuracy_2 = accuracy(ppn2.predict(X_train), y_train_02_subset)
    print("Perceptron 1 dokladosc: ", accuracy_1)
    print("Perceptron 2 dokladosc: ", accuracy_2)

    #obliczenie dokladnosci dla calego zbioru
    total_accuracy(accuracy_1, accuracy_2)

    #utworzenie klasyfikatora
    classifier = Classifier(ppn1, ppn2)

    #ustawianie zmiennych i wtorzenie grafu
    plot_decision_regions(X = X_train, y = y_train, classifier=classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    #wyswietlenie grafu
    plt.show()

def accuracy(y_results, y_train):
    return accuracy_score(y_results, y_train)

def total_accuracy(accuracy_1, accuracy_2):
    total_accuracy = (accuracy_1 + accuracy_2) / 2
    print("calkowita dokladnosc: ", total_accuracy)
    

if __name__ == '__main__':
    main()