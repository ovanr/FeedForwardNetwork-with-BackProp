from collections import namedtuple
import numpy as np

TrainResult = namedtuple('TrainResult', 'output errorRate correctCount')

class MultiLayerPerceptronException(Exception):
   pass

class IRandom():
   def makeNrandoms(self, N):
      pass

class NumpyRandom(IRandom):
   def __init__(self):
      self.rnd = np.random.default_rng()
   
   def makeNrandoms(self, N):
      return list(map(lambda x: 2*x - 1, self.rnd.random(N)))
