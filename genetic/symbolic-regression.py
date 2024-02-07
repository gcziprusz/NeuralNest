# pip install gplearn

from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
import numpy as np

def fitme(x):
  return(0.1*x*x*x + x*x)

# Training samples
X_train = np.random.uniform(-10, 10, (50,1))
y_train = [fitme(X) for X in X_train]

# Testing samples
X_test = np.random.uniform(-10, 10, (50,1))
y_test = [fitme(X) for X in X_test]

plt.scatter(X_train, y_train)
plt.title('Target distance')
plt.xlabel('angle')
plt.ylabel('distance')
plt.show()

est_gp = SymbolicRegressor(population_size=10000,parsimony_coefficient=0.1,
                           function_set=('add', 'mul'))
est_gp.fit(X_train, y_train)

X_lots = np.reshape(np.sort(np.random.uniform(-10, 10, 250)),(-1,1))

y_gp = est_gp.predict(X_lots)

plt.scatter(X_test, y_test)
plt.plot(X_lots, y_gp)
plt.title('Target distance')
plt.xlabel('angle')
plt.ylabel('distance')
plt.show()

print(est_gp._program)