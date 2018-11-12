'''
========================
Tesing the dot product:
  Loops vs Numpy
========================
'''
import numpy as np
from datetime import datetime


N = 500
V = np.random.randn(N)
W = np.random.randn(N)
def loop_dot_product(V, W):
  V_dot_W = 0
  for v, w in zip(V, W):
    V_dot_W += v*w
  return V_dot_W



Tot_time = 100000
t0 = datetime.now()
for t in range(Tot_time):
  slow_dot_product(a, b)
tf1 = datetime.now() - t0

t0 = datetime.now()
for t in range(Tot_time):
  a.dot(b)
tf2 = datetime.now() - t0

print("Loop vs Numpy, tf1/tf2:", tf1.total_seconds() / tf2.total_seconds())
