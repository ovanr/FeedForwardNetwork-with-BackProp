import numpy as np
import math
import copy
import sys
import random
import functools
import signal
from .functions import sigintHandler, sigmoid, gaussianDecay
from .objecttypes import MultiLayerPerceptronException, \
                         IRandom, \
                         NumpyRandom, \
                         TrainResult
from .constants import DECAY_RATE, \
                       OVERFIT_START, \
                       OVERFIT_THRESH

class MultiLayerPerceptronNetwork:
   _overfitConseq = -1
   _bestTestOutcome = 0
   _exitFlag = False

   def __init__(self,
                randomInterface, 
                neuronsPerLayer, 
                learningRate, 
                momentum,
                verboseFlag = False,
                outcomeCountFun = None,
                decayLearningRateFlag = False):

      self.networkWeights = []
      self.networkDeltas = []

      if not isinstance(randomInterface, IRandom):
         raise TypeError("Random generator not compatible")

      if not isinstance(neuronsPerLayer, list) \
         or len(neuronsPerLayer) <= 1:
         raise MultiLayerPerceptronException("Invalid network topology given")

      for (layerPos,noNeurons) in enumerate(neuronsPerLayer[1:]):
         # Each neuron has +1 weight to account for the bias input
         noWeights = neuronsPerLayer[layerPos] + 1

         randomNumbers = randomInterface.makeNrandoms(noWeights*noNeurons)
         layerWeights = np.array(randomNumbers, dtype="float64").reshape((noWeights, noNeurons))
         self.networkWeights.append(layerWeights)

         layerDelta = np.zeros(noNeurons)
         self.networkDeltas.append(layerDelta)

      self.tMinus1NetworkWeights = copy.deepcopy(self.networkWeights)
      self.epoch = 0
      self._initLearningRate = learningRate
      self.decayLearningRateFlag = decayLearningRateFlag
      self.momentum = momentum
      self.verboseFlag = verboseFlag

      if not outcomeCountFun:
         self.outcomeCountFun = MultiLayerPerceptronNetwork._calcOutcomesRoundEach
      else:
         self.outcomeCountFun = outcomeCountFun

      signal.signal(signal.SIGINT, functools.partial(sigintHandler, self))

   @property
   def learningRate(self):
      if self.decayLearningRateFlag:
         return gaussianDecay(self._initLearningRate, self.epoch)
      else:
         return self._initLearningRate

   @staticmethod
   def _normaliseVector(vector, vecRange):
      for c in range(vector.size):
         min = vecRange[0][c]
         max = vecRange[1][c]
         if max - min == 0:
            raise MultiLayerPerceptronException("Invalid range value 0.")

         minMax = (vector[c] - min) / (max - min)
         vector[c] = minMax

      return vector

   def _normaliseDataSet(self, dataSet):
      preparedSet = []
      noInputs = self.networkWeights[0].shape[0] - 1
      noOutputs = self.networkWeights[-1].shape[1]
      
      minNum = - 2**30
      maxNum = 2**30
  
      inRange = (np.array([maxNum] * noInputs, dtype='float64'), np.array([minNum] * noInputs, dtype='float64'))
      outRange = (np.array([maxNum] * noOutputs, dtype='float64'), np.array([minNum] * noOutputs, dtype='float64'))

      try:
         for pattern in dataSet:
            (inVec, outVec) = pattern
            self._testInputVecFormat(inVec)
            self._testOutputVecFormat(outVec)
            
            inRange = (np.minimum(inRange[0], inVec), np.maximum(inRange[1], inVec))
            outRange = (np.minimum(outRange[0], outVec), np.maximum(outRange[1], outVec))
            

      except MultiLayerPerceptronException as e:
         raise MultiLayerPerceptronException(f"Invalid train set format at pattern {pattern}") from e
      
      else:

         if self.verboseFlag:
            print("Input range detected: ", inRange)
            print("Output range detected: ", outRange)

         for pattern in dataSet:
            (inVec, outVec) = pattern
            normIn = MultiLayerPerceptronNetwork._normaliseVector(inVec, inRange)
            normOut = MultiLayerPerceptronNetwork._normaliseVector(outVec, outRange)
            preparedSet.append((normIn, normOut))

         return preparedSet

   def _testInputVecFormat(self, inputVector):
      (noWeights, noNeurons) = self.networkWeights[0].shape
      expShape = (noWeights - 1,)
      if not isinstance(inputVector, np.ndarray) \
         or inputVector.shape != expShape \
         or inputVector.dtype.name != 'float64':
            raise MultiLayerPerceptronException("Invalid input vector format")

   def _testOutputVecFormat(self, outputVector):
      (noWeights, noNeurons) = self.networkWeights[-1].shape
      expShape = (noNeurons,)
      if not isinstance(outputVector, np.ndarray) \
         or outputVector.shape != expShape \
         or outputVector.dtype.name != 'float64':
            raise MultiLayerPerceptronException("Invalid Output vector format")

   @staticmethod
   def _calcLayerOutput(inputVector, layerWeights):
      # we need to extend the vector by one to 
      # account the bias input
      inputWithBias = np.append(inputVector, np.float64(1))

      return (inputWithBias, sigmoid(inputWithBias@layerWeights))

   def _forwardPass(self, inputVector):
      outVectors = []
      for layerWeights in self.networkWeights:
         (inputWithBias, output) = self._calcLayerOutput(inputVector, layerWeights)
         outVectors.append(inputWithBias)
         inputVector = output

      outVectors.append(output)
      return outVectors

   @staticmethod
   def _calcDerivErrorSigmoidHiddenNonVec(currentNeuron, nextLayerWeights, nextLayerDeltas):
      deriv = 0
      noNeuronsNextLayer = nextLayerWeights.shape[1]
      for neuronNextId in range(noNeuronsNextLayer):
         derivErrorNetNext = nextLayerDeltas[neuronNextId]
         weight2NextNeuron = nextLayerWeights[currentNeuron][neuronNextId]
         deriv += derivErrorNetNext * weight2NextNeuron

      return deriv

   @staticmethod
   def _calcDerivErrorSigmoidHiddenVec(layerWeights, layerDeltas):
      # n1 = 1st neuron in layer
      # n1w1 = 1st weight of 1st neuron in layer
      # n1d = delta (derivErrorNet) of 1st neuron in layer
      #
      # example of layerWeights:
      # n1w1 n2w1 n3w1      
      # n1w2 n2w2 n3w2      
      # n1w3 n3w3 n3w3
      # 
      # example of layerDeltas:
      # n1d n2d n3d
      #
      # DerivErrorSigmoidHidden vector is calculated as:
      # [ Sum (n1w1*n1d, n2w1*n2d, n3w1*n3d), Sum (n1w2*n1d, n2w2*n2d, n3w2*n3d),
      #   Sum (n1w3*n3d, n3w3*n3d, n3w3*n3d) ]
      #
      # can thus be simply done by numpy's dot product
      return np.dot(layerWeights, layerDeltas)

   @staticmethod
   def _calcDerivErrorSigmoidOutput(neuronOut, expectedNeuronOut):
      return (neuronOut - expectedNeuronOut)

   @staticmethod 
   def _calcDerivErrorWeightNonVec(neuronInput, derivErrorNet):
      return neuronInput * derivErrorNet

   @staticmethod
   def _calcDerivErrorWeightVec(inputVector, derivErrorNet):
      # in1 = 1st input of all layers
      # n1d = delta of 1st neuron
      # 
      # transform input vector to form:
      # e.g
      # for input size 3 and noNeurons 2:
      # in1 in2 in3     ->    in1 in1 
      #                       in2 in2 
      #                       in3 in3
      #
      # example of derivErrorNet(layerDeltas):
      # n1d n2d
      #
      # DerivErrorWeight vector is calculated as:
      # in1*n1d in1*n2d
      # in2*n1d in2*n2d
      # in3*n1d in3*n2d
      # 
      # derivErrorNet.size == number of neurons in current layer
      noNeurons = derivErrorNet.size
      return np.tile(inputVector, (noNeurons,1)).transpose() * derivErrorNet

   def _backwardPassNonVec(self, outputVectors, expOutVector):
      # updates weights in the backwards pass (as soon as delta is calulated)

      for layerId in range(len(self.networkWeights)-1, -1, -1):
         layerWeights = self.networkWeights[layerId]
         (noWeights, noNeurons) = layerWeights.shape

         # there is 1 more output vector than there are
         # layers, since initial input vector is also present
         # so current outputVector is layerId + 1
         outVec = outputVectors[layerId+1]

         for neuron in range(noNeurons):
               neuronOut = outVec[neuron]

               derivSigmoidNet = neuronOut*(1-neuronOut)
               
               if layerId == len(self.networkWeights)-1:
                  # output layer
                  derivErrorSigmoid = MultiLayerPerceptronNetwork._calcDerivErrorSigmoidOutput(neuronOut, 
                                                                                      expOutVector[neuron])
               else:
                  # hidden layer
                  derivErrorSigmoid = MultiLayerPerceptronNetwork._calcDerivErrorSigmoidHiddenNonVec(neuron, 
                                                                                            self.networkWeights[layerId+1],
                                                                                            self.networkDeltas[layerId + 1])

               derivErrorNet = derivErrorSigmoid * derivSigmoidNet

               self.networkDeltas[layerId][neuron] = derivErrorNet

      self._forwardWeightsUpdateNonVec(outputVectors)

   def _backwardPassVec(self, outputVectors, expOutVector):
      # updates weights in the third pass

      for layerId in range(len(self.networkWeights)-1, -1, -1):
         # there is 1 more output vector than there are
         # layers (since initial input vector is also present)
         # so current outputVector is layerId + 1
         outVec = outputVectors[layerId+1]
         # prevOutVec -> layerWeights -> outVec

         derivSigmoidNet = outVec*(1-outVec)

         if layerId == len(self.networkWeights)-1:
            derivErrorSigmoid = MultiLayerPerceptronNetwork._calcDerivErrorSigmoidOutput(outVec, expOutVector)
            derivErrorNet = derivErrorSigmoid * derivSigmoidNet
         else:
            derivErrorSigmoid = MultiLayerPerceptronNetwork._calcDerivErrorSigmoidHiddenVec(self.networkWeights[layerId+1],
                                                                                   self.networkDeltas[layerId+1])
            derivErrorNet = derivErrorSigmoid * derivSigmoidNet

            # derivSigmoidNet and derivErrorSigmoid include the 
            # bias input. Thus the last element of derivErrorNet
            # will be for the bias input. It needs to be removed
            derivErrorNet = np.delete(derivErrorNet, -1, 0)

         # derivErrorNet is the 'delta'
         self.networkDeltas[layerId] = derivErrorNet
      
      self._forwardWeightsUpdateVec(outputVectors)

   def _calcNewWeight(self, curWeight, tMinus1Weight, delta, inputVal):
         derivErrorWeight = MultiLayerPerceptronNetwork._calcDerivErrorWeightNonVec(inputVal,
                                                                           delta)
         momentum = self.momentum * (curWeight - tMinus1Weight)
         newWeight = curWeight - self.learningRate*derivErrorWeight + momentum
         return newWeight
         
   def _forwardWeightsUpdateNonVec(self, outputVectors):
      for layerId in range(len(self.networkWeights)):
         layerWeights = self.networkWeights[layerId]
         layerDeltas = self.networkDeltas[layerId]
         (noWeights, noNeurons) = layerWeights.shape
         tMinus1LayerWeights = self.tMinus1NetworkWeights[layerId]
         inputVec = outputVectors[layerId]

         for neuron in range(noNeurons):
            for fromNeuron in range(noWeights):
               curWeight = layerWeights[fromNeuron][neuron]
               tMinus1Weight = tMinus1LayerWeights[fromNeuron][neuron]
               delta = layerDeltas[neuron]
               inputVal = inputVec[fromNeuron]

               newWeight = self._calcNewWeight(curWeight,
                                               tMinus1Weight,
                                               delta,
                                               inputVal)

               tMinus1LayerWeights[fromNeuron][neuron] = layerWeights[fromNeuron][neuron]
               layerWeights[fromNeuron][neuron] = newWeight

   def _forwardWeightsUpdateVec(self, outputVectors):
      # pseudo forward pass where the weights are updated using the
      # newly calculated deltas of each layer

      for layerId in range(len(self.networkWeights)):
         layerWeights = self.networkWeights[layerId]
         tMinus1LayerWeights = self.tMinus1NetworkWeights[layerId]

         # prevOutVec -> layerWeights -> outVec
         prevOutVec = outputVectors[layerId]
         
         derivErrorWeight = MultiLayerPerceptronNetwork._calcDerivErrorWeightVec(prevOutVec, 
                                                                        self.networkDeltas[layerId])

         momentum = self.momentum * (layerWeights - tMinus1LayerWeights)

         # no need to deep copy since new weights will be created from scratch
         self.tMinus1NetworkWeights[layerId] = layerWeights

         self.networkWeights[layerId] = layerWeights - self.learningRate*derivErrorWeight + momentum
   
   def _overfitCheck(self, testCorrectPercent):
      if testCorrectPercent < OVERFIT_START and self._overfitConseq == -1:
         # overfitting checks not started
         return False

      if testCorrectPercent < self._bestTestOutcome:
         self._overfitConseq += 1
         if self._overfitConseq >= OVERFIT_THRESH:
            return True
      else:
         self._overfitConseq = 0
         self._inStoreWeights = copy.deepcopy(self.networkWeights)
         self._inStoreTMinus1NetworkWeights = copy.deepcopy(self.tMinus1NetworkWeights)
         self._inStoreDeltas = copy.deepcopy(self.networkDeltas)
         self._inStoreEpoch = copy.deepcopy(self.epoch)
         self._bestTestOutcome = testCorrectPercent
         return False

   def revert2BestEpoch(self):
      if self._overfitConseq > 0:
         self.networkWeights = self._inStoreWeights
         self.networkDeltas = self._inStoreDeltas
         self.tMinus1NetworkWeights = self._inStoreTMinus1NetworkWeights
         self.epoch = self._inStoreEpoch
         return True

      return False

   @staticmethod
   def _calcErrorRate(actualVector, targetVector):
      errorRate = 0

      for (outId, output) in enumerate(actualVector):
         errorRate += (targetVector[outId] - output)**2

      errorRate /= 2
      return errorRate

   @staticmethod
   def _calcOutcomesWinnerTakeAll(actualVector, targetVector):
      if np.argmax(actualVector) == np.argmax(targetVector):
         return 1

      return 0

   @staticmethod
   def _calcOutcomesRoundEach(actualVector, targetVector):
      correctOutcomes = 0

      for (outId, output) in enumerate(actualVector):
         if targetVector[outId] == round(output):
            correctOutcomes += 1

      return correctOutcomes
 
   def trainOnPattern(self, inVec, expOutVector):
      outVectors = self._forwardPass(inVec)

      errorRate = MultiLayerPerceptronNetwork._calcErrorRate(outVectors[-1], expOutVector)
      correct = self.outcomeCountFun(outVectors[-1], expOutVector)

      self._backwardPassVec(outVectors, expOutVector)

      return TrainResult(outVectors[-1], errorRate, correct)

   def trainOnDataSet(self, trainSet, testSet, epochUntil, errorRateMin):
      if epochUntil < self.epoch:
         raise MultiLayerPerceptronException("Network is already trained.")

      self._exitFlag = False

      trainSet = self._normaliseDataSet(trainSet)
      testSet = self._normaliseDataSet(testSet)

      errorRateVec = []
      outcomeVec = []

      for _ in range(epochUntil - self.epoch):
         trainErrorRate = 0
         correctOutcomes = 0

         for (inVec, expOutVector) in trainSet:
            res = self.trainOnPattern(inVec, expOutVector)
            trainErrorRate += res.errorRate
            correctOutcomes += res.correctCount
            if self.verboseFlag:
               print("Train Input:", inVec, " Output:", res.output, " Error: ", res.errorRate)

         trainErrorRate /= len(trainSet)
         trainCorrectPercent = 100 * correctOutcomes / len(trainSet)

         (testCorrectPercent, testErrorRate) = self._evaluateOnDataSet(testSet)
         
         errorRateVec.append((self.epoch, trainErrorRate, testErrorRate))
         outcomeVec.append((self.epoch, trainCorrectPercent, testCorrectPercent))

         print(f"Epoch {self.epoch},",
               f"Learning Rate {self.learningRate:.5f},",
               f"Train Error Rate {trainErrorRate:.5f},",
               f"Test Error Rate {testErrorRate:.5f}")

         print(f"Epoch {self.epoch},",
               f"Train Correct outcomes {trainCorrectPercent:.5f}%,",
               f"Test Correct outcomes {testCorrectPercent:.5f}%")

         if self._overfitCheck(testCorrectPercent):
            print("Overfitting detected, ending training...")
            break

         if errorRateMin >= trainErrorRate:
            print("Reached ideal error rate. ending training...")
            break

         if self._exitFlag:
            print("User specified exit...")
            break
         
         self.epoch += 1

      print("Training stopped.")
      
      if self.revert2BestEpoch():
         print("Reverted network back to best epoch ", self.epoch)

      with open("error.txt", "w+") as f:
         for (epoch, trainErr, testErr) in errorRateVec:
            f.write("{:d} {:.4f} {:.4f}\n".format(epoch, trainErr, testErr))

      with open("success.txt", "w+") as f:
         for (epoch, trainPer, testPer) in outcomeVec:
            f.write("{:d} {:.2f} {:.2f}\n".format(epoch, trainPer, testPer))   

   def _evaluateOnDataSet(self, testSet):
      globalErrorRate = 0
      correctOutcomes = 0

      for (inVec, expOutVector) in testSet:
         outVectors = self._forwardPass(inVec)
         errorRate = MultiLayerPerceptronNetwork._calcErrorRate(outVectors[-1], expOutVector)
         correct = self.outcomeCountFun(outVectors[-1], expOutVector)
         
         if self.verboseFlag:
            print("Test Input:", inVec, " Output:", outVectors[-1], " Error: ", errorRate)

         globalErrorRate += errorRate
         correctOutcomes += correct
      
      correctPercent = 100*correctOutcomes/len(testSet)
      avgErrorRate = globalErrorRate / len(testSet)

      return (correctPercent, avgErrorRate)

   def evaluateOnDataSet(self, testSet):
      self._exitFlag = False

      testSet = self._normaliseDataSet(testSet)
      self._evaluateOnDataSet(testSet)