import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

def f_lagrange(x, X, Y):
  n = len(X) - 1
  return sum((Y[k] * L(k, x, X)) for k in range(n + 1))
  # val = 0
  # for k in range(n + 1):
  #  # print(Y[k], L(x, X, k, n), Y[k] * L(x, X, k, n))
  #   val += Y[k] * L(x, X, k, n)
  # return val

def L(k, x, X):
  n = len(X) - 1
  return np.prod([(x - X[j]) / (X[k] - X[j]) for j in range(n + 1) if k != j])
  # prod = 1
  # for j in range(n + 1):
  #   if X[j] == X[k]:
  #     continue
  #   prod *= (x - X[j]) / (X[k] - X[j])
  # return prod

def auc_trapezoidal(X, Y, precision, f):
  a = X[0]
  b = X[-1]
  h = (b - a) / precision
  somatorio = 0
  for i in range(precision + 1):
    x_i = a + (i * h)
    f_x_i = f(x_i, X, Y)
    # print('x:', x_i, 'f(x):', f_x_i)
    if i > 0 and i < precision:
      somatorio += 2 * f_x_i
    else:
      somatorio += f_x_i
  return (h / 2) * somatorio

def auc_boole(X, Y, f):
  a = X[0]
  b = X[-1]
  h = (b - a) / 4
  coeficients = [7, 32, 12, 32, 7]
  for i, coef in enumerate(coeficients):
    f_x_i = a + h * i
    somatorio += coef * f(f_x_i, X, Y)
  return 2 * h * somatorio / 45


def main():
  np.random.seed(1)
  # X = [93.3, 98.9, 104.4]
  # Y = [1548, 1544, 1538]
  # print(f(100, X, Y))

  size = 100

  X = sorted(np.unique(np.random.uniform(size=size)))
  Y = sorted(np.random.uniform(size=len(X)))

  print(auc_trapezoidal(X, Y, 50, f_lagrange), auc_boole(X, Y, f_lagrange))

  plt.figure()
  lw = 2
  plt.plot(X, Y, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % auc(X, Y))
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0005])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()

if __name__ == "__main__":
  main()