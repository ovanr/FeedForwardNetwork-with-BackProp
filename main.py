from mlpBp.parser import readParams, parseParams
from mlpBp.mlp import MultiLayerPerceptronNetwork, NumpyRandom
from sys import argv

params = parseParams(readParams(argv[1]))
rndGen = NumpyRandom()
mlp = MultiLayerPerceptronNetwork(rndGen,
                         params["layout"], 
                         params["learningRate"], 
                         params["momentum"],
                         params["verbose"],
                         params["outcomeCalcMethod"],
                         params["decayLearningRate"])

mlp.trainOnDataSet(params["trainSet"], 
                   params["testSet"], 
                   params["maxIterations"], 
                   params["minErrorRate"])

# mlp.evaluate(params["testSet"])
