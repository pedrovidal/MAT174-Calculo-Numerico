import sys
import numpy as np
import scipy
import bisect
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

def f_lagrange(x, X, Y):
  # print(X, Y)
  def L(k, x, X):
    n = len(X) - 1
    return np.prod([(x - X[j]) / (X[k] - X[j]) for j in range(n + 1) if k != j])
  n = len(X) - 1
  return sum((Y[k] * L(k, x, X)) for k in range(n + 1))

def auc_trapezoidal(X, Y, precision, f):
  # print('trapezoidal:')
  a = X[0]
  b = X[-1]
  h = (b - a) / precision
  somatorio = 0
  X_trap = []
  Y_trap = []
  for i in range(precision + 1):
    x_i = a + (i * h)
    ind = bisect.bisect_left(X, x_i)
    f_x_i = f(x_i, X[max(0, ind - 2): min(ind + 2, len(X))], Y[max(0, ind - 2): min(ind + 2, len(X))])
    # print('x:', x_i, 'f(x):', f_x_i)
    X_trap.append(x_i)
    Y_trap.append(f_x_i)

    if i > 0 and i < precision:
      somatorio += 2 * f_x_i
    else:
      somatorio += f_x_i
  return X_trap, Y_trap, (h / 2) * somatorio

def auc_boole(X, Y, a, b, f):
  # print('boole:')
  h = (b - a) / 4
  coeficients = [7, 32, 12, 32, 7]
  somatorio = 0
  X_boole = []
  Y_boole = []
  for i, coef in enumerate(coeficients):
    x_i = a + (i * h)
    ind = bisect.bisect_left(X, x_i)
    f_x_i = f(x_i, X[max(0, ind - 2): min(ind + 2, len(X))], Y[max(0, ind - 2): min(ind + 2, len(X))])
    # print('x:', x_i, 'f(x):', f_x_i)
    X_boole.append(x_i)
    Y_boole.append(f_x_i)
    somatorio += coef * f_x_i
  return X_boole, Y_boole, 2 * h * somatorio / 45

def auc_simpson(X, Y, precision, f):
  # print('simpson:')
  a = X[0]
  b = X[-1]
  h = (b - a) / precision
  somatorio = 0
  X_simpson = []
  Y_simpson = []
  for i in range(precision + 1):
    x_i = a + (i * h)
    ind = bisect.bisect_left(X, x_i)
    f_x_i = f(x_i, X[max(0, ind - 2): min(ind + 2, len(X))], Y[max(0, ind - 2): min(ind + 2, len(X))])
    # print('x:', x_i, 'f(x):', f_x_i)
    if i == 0:
      X_simpson.append(X[0])
      Y_simpson.append(Y[0])
      somatorio += Y[0]
    elif i == precision:
      X_simpson.append(X[-1])
      Y_simpson.append(Y[-1])
      somatorio += Y[-1]
    elif i % 2:
      X_simpson.append(x_i)
      Y_simpson.append(f_x_i)
      somatorio += 4 * f_x_i
    else:
      X_simpson.append(x_i)
      Y_simpson.append(f_x_i)
      somatorio += 2 * f_x_i
  return X_simpson, Y_simpson, h * somatorio / 3

def main():
  # np.random.seed(3)
  np.random.seed(9)
  # X = [93.3, 98.9, 104.4]
  # Y = [1548, 1544, 1538]
  # print(f(100, X, Y))

  size = 98

  X = sorted(np.unique(np.random.uniform(size=size)))
  Y = sorted(np.random.uniform(size=len(X)))

  X = np.append(X, (0, 1))
  Y = np.append(Y, (np.random.uniform(size=1), np.random.uniform(size=1)))

  X = sorted(X)
  Y = sorted(Y)

  # print(X, Y)

  precision = 20
  
  X_trap, Y_trap, auc_trap = auc_trapezoidal(X, Y, precision, f_lagrange)
  X_simpson, Y_simpson, auc_simp = auc_simpson(X, Y, precision, f_lagrange)

  X_boole, Y_boole, auc_boo = auc_boole(X, Y, 0.0, 0.25, f_lagrange)
  
  X_boole2, Y_boole2, auc_boo2 = auc_boole(X, Y, 0.25, 0.5, f_lagrange)
  X_boole.extend(X_boole2)
  Y_boole.extend(Y_boole2)
  auc_boo += auc_boo2
  
  X_boole2, Y_boole2, auc_boo2 = auc_boole(X, Y, 0.5, 0.75, f_lagrange)
  X_boole.extend(X_boole2)
  Y_boole.extend(Y_boole2)
  auc_boo += auc_boo2
  
  X_boole2, Y_boole2, auc_boo2 = auc_boole(X, Y, 0.75, 1.0, f_lagrange)
  X_boole.extend(X_boole2)
  Y_boole.extend(Y_boole2)
  auc_boo += auc_boo2

  plt.figure()
  lw = 2
  plt.plot(X, Y, color='darkorange',
           lw=lw, label='original ROC curve (area = %0.4f)' % auc(X, Y))
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

  plt.plot(X_trap, Y_trap, color='darkblue',
           lw=lw, label='trapezoidal ROC curve (area = %0.4f)' % auc_trap)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0005])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()


  plt.figure()
  lw = 2
  plt.plot(X, Y, color='darkorange',
           lw=lw, label='original ROC curve (area = %0.4f)' % auc(X, Y))
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

  plt.plot(X_boole, Y_boole, color='darkred',
           lw=lw, label='boole ROC curve (area = %0.4f)' % auc_boo)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0005])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()


  plt.figure()
  lw = 2
  plt.plot(X, Y, color='darkorange',
           lw=lw, label='original ROC curve (area = %0.4f)' % auc(X, Y))
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

  plt.plot(X_simpson, Y_simpson, color='yellow',
           lw=lw, label='simpson ROC curve (area = %0.4f)' % auc_simp)

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0005])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()



if __name__ == "__main__":
  main()