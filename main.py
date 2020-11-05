from ffnBp.parser import readParams, parseParams
from ffnBp.ffn import FeedForwardNetwork, NumpyRandom
from sys import argv

params = parseParams(readParams(argv[1]))
rndGen = NumpyRandom()
ffn = FeedForwardNetwork(rndGen,
                         params["layout"], 
                         params["learningRate"], 
                         params["momentum"],
                         params["verbose"],
                         params["outcomeCalcMethod"],
                         params["decayLearningRate"])

ffn.trainOnDataSet(params["trainSet"], 
                   params["testSet"], 
                   params["maxIterations"], 
                   params["minErrorRate"])

# ffn.evaluate(params["testSet"])
