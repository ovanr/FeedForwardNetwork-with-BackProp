import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from hypothesis import given, example
from hypothesis.strategies import text, floats
from hypothesis.extra.numpy import arrays
import copy

from mlpBp.mlp import MultiLayerPerceptronNetwork, \
                      IRandom, \
                      NumpyRandom, \
                      sigmoid, \
                      OVERFIT_THRESH, \
                      OVERFIT_START, \
                      DECAY_RATE
from mlpBp.functions import gaussianDecay
from unittest.mock import Mock, \
                          patch, \
                          MagicMock

# to run this test do
# cd 'mlpBp' and run: python -m unittest tests.test_mlp -v

# TODO: make some tests work without relying on other units 
# (substitute dependent method calls with Mocks)

TEST_NETWORK_TOPO = [3,2,1]

class TestConstRandom(IRandom):
    def __init__(self, const):
        self.const = const

    def makeNrandoms(self, N):
        return [self.const]*N

class TestSeqRandom(IRandom):
    def makeNrandoms(self, N):
        return list(range(1,N+1))
        
class TestMLPMethods(unittest.TestCase):
    def testNumpyRandomGeneratesWithinRange(self):
        ns = NumpyRandom().makeNrandoms(10)
        self.assertEqual(10, len(ns))

        prev = -2
        for n in ns:
            self.assertIsInstance(n, float)
            self.assertGreaterEqual(n, -1)
            self.assertLessEqual(n, 1)
            self.assertNotAlmostEqual(n, prev, places=5)
            prev = n

    @given(out1=arrays('float64', (4,), elements=floats(-1,1)),
           out2=arrays('float64', (5,), elements=floats(-1,1)),
           out3=arrays('float64', (3,), elements=floats(-1,1)),
           expOut=arrays('float64', (3,), elements=floats(-1,1)))
    def testBackwardPassNonVecEqualsVecVersion(self, out1, out2, out3, expOut):
        testRandom = TestSeqRandom()
        testVecMLP = MultiLayerPerceptronNetwork(testRandom, [3,4,3], 0.5, 0.6)
        testNonVecMLP = MultiLayerPerceptronNetwork(testRandom, [3,4,3], 0.5, 0.6)
        
        outputVectors = [out1, out2, out3]

        testNonVecMLP._backwardPassNonVec(outputVectors, expOut)
        testVecMLP._backwardPassVec(outputVectors, expOut)

        for i in range(len(testVecMLP.networkWeights)):
            assert_array_equal(testVecMLP.networkWeights[i], testNonVecMLP.networkWeights[i])

    @given(out1=arrays('float64', (4,), elements=floats(-1,1, width=16), unique=True),
           out2=arrays('float64', (5,), elements=floats(-1,1, width=16), unique=True),
           out3=arrays('float64', (3,), elements=floats(-1,1, width=16), unique=True),
           delta1=arrays('float64', (4,), elements=floats(-1,1, width=16), unique=True),
           delta2=arrays('float64', (3,), elements=floats(-1,1, width=16), unique=True))
    def testForwardWeightUpdateNonVecEqualsVecVersion(self, out1, out2, out3, delta1, delta2):
        testRandom = TestSeqRandom()
        testVecMLP = MultiLayerPerceptronNetwork(testRandom, [3,4,3], 0.5, 0.6)
        testNonVecMLP = MultiLayerPerceptronNetwork(testRandom, [3,4,3], 0.5, 0.6)
        
        prevWeights = copy.deepcopy(testVecMLP.networkWeights)
        testVecMLP.networkDeltas = [delta1, delta2]
        testNonVecMLP.networkDeltas = [delta1, delta2]
        outputVectors = [out1, out2, out3]

        testNonVecMLP._forwardWeightsUpdateNonVec(outputVectors)
        testVecMLP._forwardWeightsUpdateVec(outputVectors)

        for i in range(len(testVecMLP.networkWeights)):
            assert_array_equal(testVecMLP.networkWeights[i], testNonVecMLP.networkWeights[i])
            assert_raises(AssertionError, assert_array_equal, testVecMLP.networkWeights[i], prevWeights[i])

    def testCorrectLayerDimentions(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        self.assertEqual(len(TEST_NETWORK_TOPO)-1, len(testMLP.networkWeights))

        for id,layer in enumerate(testMLP.networkWeights):
            (noWeights,noNeurons) = layer.shape
            self.assertEqual(noWeights,TEST_NETWORK_TOPO[id] + 1)
            self.assertEqual(noNeurons,TEST_NETWORK_TOPO[id+1])

    def testCorrectDeltaDimentions(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        self.assertEqual(len(TEST_NETWORK_TOPO)-1, len(testMLP.networkDeltas))

        for id,layer in enumerate(testMLP.networkDeltas):
            self.assertEqual(1, layer.ndim)
            self.assertEqual(layer.size, TEST_NETWORK_TOPO[id+1])

    def testCorrectNormalisation(self):
        vec = np.array([150, 150, 150], dtype='float64')
        actVec = MultiLayerPerceptronNetwork._normaliseVector(vec, (np.array([50,0, 50], dtype='float64'), np.array([450, 400, 450], dtype='float64')))
        expVec = np.array([0.25, 0.375, 0.25], dtype='float64')
        self.assertTrue(np.array_equal(actVec, expVec))

    def testRaisesInvalidRandomGenerator(self):
        invalidRandom = list()
        self.assertRaises(TypeError, MultiLayerPerceptronNetwork, invalidRandom, TEST_NETWORK_TOPO, 0, 0)

    def testRaisesInvalidTopology(self):
        invalidTopo = 5
        testRandom = TestConstRandom(1)
        self.assertRaisesRegex(Exception, "Invalid network topology given", MultiLayerPerceptronNetwork, testRandom, invalidTopo, 0, 0)

    def testCorrectInitializedWeights(self):
        testRandom = TestSeqRandom()
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)
        for layer in testMLP.networkWeights:
            expect = 1
            for x in layer:
                for y in x:
                    self.assertEqual(expect, y)
                    expect += 1

    def testCorrectDataSetNormalisation(self):
        testMLP = Mock(wraps=MultiLayerPerceptronNetwork, autospec=True)
        testMLP.networkWeights = MagicMock()
        testMLP.networkWeights = [Mock(), Mock()]
        testMLP.networkWeights[0].shape =  [4]
        testMLP.networkWeights[-1].shape = [0,2]
        testMLP.verboseFlag = False
        
        testMLP._testInputVecFormat = lambda x:x
        testMLP._testOutputVecFormat = lambda x:x

        inVec = [np.array([1, 1, 2], dtype='float64'),
                 np.array([6, 1, 1], dtype='float64'),
                 np.array([4, 2, 1], dtype='float64'),
                 np.array([6, 1, 0], dtype='float64')]

        outVec = [np.array([2, 4], dtype='float64'),
                  np.array([2, 3], dtype='float64'),
                  np.array([2, 2], dtype='float64'),
                  np.array([3, 1], dtype='float64')]


        testDataSet = [(inVec[c], outVec[c]) for c in range(len(inVec))]
        
        testMLP._normaliseDataSet(testMLP, testDataSet)

    def testAcceptsCorrectInputVector(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        try:
            inputVector = np.array([1] * TEST_NETWORK_TOPO[0], dtype='float64')
            testMLP._testInputVecFormat(inputVector)

        except Exception:
            self.fail("Raised exception although input is valid")

    def testRaisesInvalidInputVector(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        inputVector = np.array([1] * (TEST_NETWORK_TOPO[0] + 1), dtype='float64')
        self.assertRaisesRegex(Exception, "Invalid input vector format", testMLP._testInputVecFormat, inputVector)

        inputVector = 5
        self.assertRaisesRegex(Exception, "Invalid input vector format", testMLP._testInputVecFormat, inputVector)

    def testAcceptsCorrectOutputVector(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)
        try:
            outputVector = np.array([1] * TEST_NETWORK_TOPO[-1], dtype='float64')
            testMLP._testOutputVecFormat(outputVector)

        except Exception:
            self.fail("Raised exception although output is valid")

    def testRaisesInvalidOutputVector(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        outVector = np.array([1] * (TEST_NETWORK_TOPO[-1] + 1), dtype='float64')
        self.assertRaisesRegex(Exception, "Invalid Output vector format", testMLP._testOutputVecFormat, outVector)

        outVector = 5
        self.assertRaisesRegex(Exception, "Invalid Output vector format", testMLP._testOutputVecFormat, outVector)

    def testCorrectLayerOutputCalculation(self):
       
        inputVector = np.array([1, 1], dtype='float64')
        layerWeights = np.ones((3, 2), dtype='float64')

        (newInput, output) = MultiLayerPerceptronNetwork._calcLayerOutput(inputVector, layerWeights)
       
        expectInput = sigmoid(np.array([3, 3], dtype='float64'))
        
        self.assertTrue(np.array_equal(expectInput, output))
        
        self.assertTrue(np.array_equal(newInput, np.array([1] * 3, dtype='float64')))

    def testCorrectForwardPass(self):
        tests = [([3,3,3], [13,13,13]),
                 ([4,1], [5])]

        for (layout, expOut) in tests:
            with self.subTest(test=(layout, expOut)):
                testRandom = TestConstRandom(1)
                testMLP = MultiLayerPerceptronNetwork(testRandom, layout, 0, 0)

                inputVec = np.ones(layout[0], dtype='float64')
                with patch('mlpBp.mlp.sigmoid', lambda x: x) as _:
                    outputs = testMLP._forwardPass(inputVec)

                    expArray = np.array(expOut, dtype='float64')

                    self.assertTrue(np.array_equal(expArray, outputs[-1]))

    def testCorrectDerivErrorSigmoidHiddenNonVec(self):
        tests = [([2,2,2], [[1,1,1]], 0, 6),
                 ([3,3,3],[[0,0,0],[2,2,2]], 1, 18)]

        for (deltas,weights, curNode, expOut) in tests:
            with self.subTest(test=(deltas,weights,expOut)):
                nextLayerDeltas = np.array(deltas, dtype='float64')
                nextLayerWeights= np.array(weights, dtype='float64')

                result = MultiLayerPerceptronNetwork._calcDerivErrorSigmoidHiddenNonVec(curNode, nextLayerWeights, nextLayerDeltas)

                self.assertEqual(result, expOut)

    def testCorrectBackwardPassNonVec(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, [3,3,3], learningRate = 0.5, momentum = 0)
        
        expOutVector = MagicMock(spec=np.ndarray)

        outputVectors = [np.array([2,2,2,2]),
                         np.array([2,2,2,2]),
                         np.array([2,2,2])]

        with patch('mlpBp.mlp.MultiLayerPerceptronNetwork._calcDerivErrorSigmoidOutput', lambda _,__: 1):
            with patch('mlpBp.mlp.MultiLayerPerceptronNetwork._calcDerivErrorSigmoidHiddenNonVec', lambda _,__,___: 2):
                with patch('mlpBp.mlp.MultiLayerPerceptronNetwork._forwardWeightsUpdateNonVec', lambda _,__:_):
                    testMLP._backwardPassNonVec(outputVectors, expOutVector)
                    for layerId in range(len(testMLP.networkDeltas)):
                        if layerId == len(testMLP.networkDeltas)-1:
                            self.assertTrue(np.alltrue(testMLP.networkDeltas[layerId] == -2))
                        else:
                            self.assertTrue(np.alltrue(testMLP.networkDeltas[layerId] == -4))

    def testCorrectForwardWeightsUpdateNonVec(self):
        testRandom = TestConstRandom(1)
        testMLP = MultiLayerPerceptronNetwork(testRandom, [3,3,3], learningRate = 0.5, momentum = 0)

        outputVectors = MagicMock(spec=list)

        with patch('mlpBp.mlp.MultiLayerPerceptronNetwork._calcDerivErrorWeightNonVec', lambda _,__: 1):
            testMLP._forwardWeightsUpdateNonVec(outputVectors)
            for layerId in range(len(testMLP.networkWeights)):
                self.assertTrue(np.alltrue(testMLP.networkWeights[layerId] == 0.5))
                self.assertTrue(np.alltrue(testMLP.tMinus1NetworkWeights[layerId] == 1))

    def testCorrectErrorRate(self):
        tests = [([1,0,2], [1,0,0], 2),
                 ([1,0,1], [1,0,1], 0),
                 ([1,1], [0,0], 1)
                ]
        
        for (actual,target, expErr) in tests:
            with self.subTest(test= (actual,target, expErr)):
                out1 = np.array(actual, dtype='float64')
                out2 = np.array(target, dtype='float64')

                err = MultiLayerPerceptronNetwork._calcErrorRate(out1, out2)
                self.assertEqual(err, expErr)
    
    def testCorrectCalcOutcomeRoundEach(self):
        tests = [([1,0,2], [1,0,0], 2),
                 ([1,0,1], [1,0,1], 3),
                 ([1,1], [0,0], 0)
                ]
        
        for (actual,target, expOutcome) in tests:
            with self.subTest(test=(actual,target, expOutcome)):
                out1 = np.array(actual, dtype='float64')
                out2 = np.array(target, dtype='float64')

                out = MultiLayerPerceptronNetwork._calcOutcomesRoundEach(out1, out2)
                self.assertEqual(out, expOutcome)

    def testCorrectCalcOutcomeWinnerTakeAll(self):
        tests = [([1,0,2], [1,0,0], 0),
                 ([0.98,0,0.95], [1,0,0], 1),
                 ([0,0], [0,0], 1)
                ]
        
        for (actual,target, expOutcome) in tests:
            with self.subTest(test=(actual,target, expOutcome)):
                out1 = np.array(actual, dtype='float64')
                out2 = np.array(target, dtype='float64')

                out = MultiLayerPerceptronNetwork._calcOutcomesWinnerTakeAll(out1, out2)
                self.assertEqual(out, expOutcome)

    def testCorrectOverfitDetection(self):
        mockMLP = Mock(wraps=MultiLayerPerceptronNetwork)
        mockMLP._overfitConseq = OVERFIT_THRESH - 1
        mockMLP._bestTestOutcome = OVERFIT_START + 1
        mockMLP._inStoreWeights = Mock()
        mockMLP._inStoreTMinus1NetworkWeights = Mock()
        mockMLP._inStoreDeltas = Mock()
        mockMLP._inStoreEpoch = Mock()
        self.assertTrue(mockMLP._overfitCheck(mockMLP, OVERFIT_START - 1))
        self.assertEqual(mockMLP._overfitConseq, OVERFIT_THRESH)

    def testCorrectOverfitBreak(self):
        mockMLP = Mock(wraps=MultiLayerPerceptronNetwork)
        mockMLP._overfitConseq = 30
        mockMLP._bestTestOutcome = OVERFIT_START + 1
        mockMLP.networkWeights = Mock()
        mockMLP.tMinus1NetworkWeights = Mock()
        mockMLP.networkDeltas = Mock()
        mockMLP.epoch = Mock()
        self.assertFalse(mockMLP._overfitCheck(mockMLP, OVERFIT_START + 2))
        self.assertEqual(mockMLP._overfitConseq, 0)
        self.assertEqual(mockMLP._bestTestOutcome, OVERFIT_START + 2)

    def testCorrectEpochReverting(self):
        mockMLP = Mock(wraps=MultiLayerPerceptronNetwork)
        mockMLP._overfitConseq = 30
        mockMLP._inStoreWeights = Mock()
        mockMLP._inStoreTMinus1NetworkWeights = Mock()
        mockMLP._inStoreDeltas = Mock()
        mockMLP._inStoreEpoch = Mock()
        mockMLP.revert2BestEpoch(mockMLP)
        self.assertEqual(mockMLP._inStoreWeights, mockMLP.networkWeights)

    def testLearningRateDecay(self):
        testRandom = TestSeqRandom()
        testMLP = MultiLayerPerceptronNetwork(testRandom, 
                                     neuronsPerLayer=[3,4,3], 
                                     learningRate=0.5, 
                                     momentum=0.6,
                                     decayLearningRateFlag=True)

        testMLP.epoch = 90

        adjustedRate = testMLP.learningRate
        self.assertEqual(adjustedRate, gaussianDecay(0.5,90) )