import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file = np.loadtxt("/content/dane1.txt")
X = file[:,[0]]
y = file[:,[1]]

plt.plot(X, y, 'g*')
plt.show

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, '*r')
plt.show

#MODEL1 - liniowy model parametryczny => y = ax + b

F1 = np.hstack([X_train, np.ones(X_train.shape)]) #macierz parametrów
V1 = np.linalg.inv(F1.T @ F1) @ F1.T @ y_train #macierz psełdo odwrotna mnozona przez y

#TRAIN
#liczenie bledu 
E1_train = y_train - (V1[0]*X_train + V1[1])
#sredni blad kwadratowy
MSE1_train = (E1_train.T @ E1_train)/len(E1_train)
print(MSE1_train)

#TEST
#liczenie bledu 
E1_test = y_test - (V1[0]*X_test + V1[1])
#sredni blad kwadratowy
MSE1_test = (E1_test.T @ E1_test)/len(E1_test)
print(MSE1_test)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, '*r')
plt.plot(X, V1[0]*X + V1[1])
plt.show

#MODEL 2 - kwadratowy => y = ax^2 + bx + c

F2 = np.hstack([X_train**2,X_train, np.ones(X_train.shape)]) #macierz parametrów
V2 = np.linalg.pinv(F2) @ y_train #macierz psełdo odwrotna mnozona przez y 

# TRAIN
# liczenie bledu 
E2_train = y_train - (V2[0]*np.square(X_train) + V2[1]*X_train + V2[2])
# sredni blad kwadratowy
MSE2_train = (E2_train.T @ E2_train)/len(E2_train)
print(MSE2_train)

# TEST
# liczenie bledu 
E2_test = y_test - (V2[0]*np.square(X_test) + V2[1]*X_test + V2[2])
# sredni blad kwadratowy
MSE2_test = (E2_test.T @ E2_test)/len(E2_test)
print(MSE2_test)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, '*r')
plt.plot(X, V1[0]*X + V1[1]) #model 1
plt.plot(X, V2[0]*X*X + V2[1]*X, V2[2]) #model 2
plt.show