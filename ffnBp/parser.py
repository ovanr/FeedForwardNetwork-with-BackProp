import numpy as np
from .ffn import FeedForwardNetwork
import math
import random

def readDataSet(fileName):
   with open(fileName, "r") as f:
      ls = f.read().split("\n")
      return [list(filter(lambda x: x, l.split(" "))) for l in filter(lambda x: x, ls)]

def parseDataSet(rawData, noInputs, noOutputs):
   dataSet = []

   for record in rawData:
      inputs = record[0:noInputs]
      outputs = record[noInputs: noOutputs+noInputs]
      dataSet.append((np.array(inputs, dtype="float64"), np.array(outputs, dtype="float64")))
   
   return dataSet

def splitDataSet(dataSet):
   noRecords = len(dataSet)
   splittedSet = {}
   trainSet = []
   testSet = []

   for pattern in dataSet:
      (_, outVec) = pattern
      outSet = np.argmax(outVec)
      if outSet not in splittedSet:
         splittedSet[outSet] = []
      splittedSet[outSet].append(pattern)
   
   noOutputs = len(splittedSet.keys())

   for _ in range(0,math.floor(0.7*noRecords/noOutputs)):
      for k in splittedSet.keys():
         if len(splittedSet[k]) > 0:
            pattern = splittedSet[k].pop()
            trainSet.append(pattern)

   rest = splittedSet.values()
   for r in rest:
      testSet.extend(r)

   random.shuffle(trainSet)
   random.shuffle(testSet)
   
   alls = trainSet + testSet
   assert len(alls) == len(dataSet)
   return (trainSet, testSet)

def readParams(fileName):
   params = {}

   with open(fileName, "r") as f:
      for l in f.read().split("\n"):
         if not l:
            continue
         if l[0] == '#':
            continue
         (name, value) = list(filter(lambda x: x, l.split(" ")))
         params[name] = value
      return params

def getLayout(params):
   layout = [int(params["numInputNeurons"])]

   layout.append(int(params.get("numHiddenLayerOneNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerTwoNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerThreeNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerFourNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerFiveNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerSixNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerSevenNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerEightNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerNineNeurons", 0)))
   layout.append(int(params.get("numHiddenLayerTenNeurons", 0)))
   
   layout.append(int(params["numOutputNeurons"]))

   return list(filter(lambda x: x > 0, layout))

def getOutcomeCalcMethod(params):
   if "outcomeCalcMethod" not in params:
      return None

   if params["outcomeCalcMethod"] == 'winner-take-all':
      return FeedForwardNetwork._calcOutcomesWinnerTakeAll
   elif params["outcomeCalcMethod"] == 'round-each':
      return FeedForwardNetwork._calcOutcomesRoundEach
   else:
      return None

def parseParams(params):
   params["learningRate"] = float(params.get("learningRate", 0.5))
   params["decayLearningRate"] = True if "decayLearningRate" in params and \
                                          params["decayLearningRate"] == "True" \
                                      else False
                                        
   params["momentum"] = float(params.get("momentum", 0.6))
   params["maxIterations"] = int(params.get("maxIterations", 2000))
   params["minErrorRate"] = float(params.get("minErrorRate", 0.0001))
   params["verbose"] = True if "verbose" in params and \
                               params["verbose"] == "True" \
                            else False

   params["outcomeCalcMethod"] = getOutcomeCalcMethod(params)

   params["layout"] = getLayout(params)
   
   if "trainFile" in params:
      trainSet = readDataSet(params["trainFile"])
      del params["trainFile"]
      params["trainSet"] = parseDataSet(trainSet, 
                                        int(params["numInputNeurons"]), 
                                        int(params["numOutputNeurons"]))
   
   if "testFile" in params:
      testSet = readDataSet(params["testFile"])
      del params["testFile"]
      params["testSet"] = parseDataSet(testSet, 
                                       int(params["numInputNeurons"]), 
                                       int(params["numOutputNeurons"]))
   
   if "dataFile" in params:
      dataSet = readDataSet(params["dataFile"])
      del params["dataFile"]
      dataSet = parseDataSet(dataSet, 
                             int(params["numInputNeurons"]), 
                             int(params["numOutputNeurons"]))
      (trainSet, testSet) = splitDataSet(dataSet)
      params["trainSet"] = trainSet
      params["testSet"] = testSet

   return params
