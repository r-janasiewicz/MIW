from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
        
iris = datasets.load_iris()
X = iris.data[:, [1,2]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print(y)

entropy = tree.DecisionTreeClassifier(criterion='entropy',random_state=1) # Uzupełnić parametry konstruktora
entropy = entropy.fit(X, y)
y_pred_entropy = entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

tree.plot_tree(entropy) 
plt.show()

gini = tree.DecisionTreeClassifier(criterion='gini',random_state=1,splitter='best') # Uzupełnić parametry konstruktora
gini = gini.fit(X, y)
y_pred_gini = gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)
tree.plot_tree(gini) 
plt.show()

print("Dokładność drzewa dla entropii:", accuracy_entropy)
print("Dokładność drzewa dla współczynnika Giniego:", accuracy_gini)

dat = export_graphviz(entropy, out_file=None,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dat)  
graph.write_png('diabetes3.png')
Image(graph.create_png())

plot_decision_regions(X=X, y=y, classifier=gini)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X=X, y=y, classifier=entropy)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()

# Ustawienie różnych głębokości drzewa i trening modelu
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_scores = []
test_scores = []

for depth in depths:
    model = DecisionTreeClassifier(criterion="entropy",max_depth=depth)
    model.fit(X_train, y_train)
    
    # Sprawdzenie dokładności na zbiorze treningowym i testowym
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Wizualizacja wyników
plt.plot(depths, train_scores, label='Training accuracy')
plt.plot(depths, test_scores, label='Test accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Ustawienie różnej liczby drzew i trening modelu
n_estimators = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
train_scores = []
test_scores = []

for n in n_estimators:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    
    # Sprawdzenie dokładności na zbiorze treningowym i testowym
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Wizualizacja wyników
import matplotlib.pyplot as plt

plt.plot(n_estimators, train_scores, label='Train accuracy')
plt.plot(n_estimators, test_scores, label='Test accuracy')
plt.xscale('log')
plt.xlabel('Liczba drzew')
plt.ylabel('Accuracy')
plt.legend()
plt.show()