import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from hypothesis import given, example
from hypothesis.strategies import text, floats
from hypothesis.extra.numpy import arrays
import copy

from ffnBp.ffn import FeedForwardNetwork, \
                      IRandom, \
                      NumpyRandom, \
                      sigmoid, \
                      OVERFIT_THRESH, \
                      OVERFIT_START, \
                      DECAY_RATE
from ffnBp.functions import gaussianDecay
from unittest.mock import Mock, \
                          patch, \
                          MagicMock

# to run this test do
# cd 'ffnBp' and run: python -m unittest tests.test_ffn -v

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
        
class TestFFNMethods(unittest.TestCase):
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
        testVecFFN = FeedForwardNetwork(testRandom, [3,4,3], 0.5, 0.6)
        testNonVecFFN = FeedForwardNetwork(testRandom, [3,4,3], 0.5, 0.6)
        
        outputVectors = [out1, out2, out3]

        testNonVecFFN._backwardPassNonVec(outputVectors, expOut)
        testVecFFN._backwardPassVec(outputVectors, expOut)

        for i in range(len(testVecFFN.networkWeights)):
            assert_array_equal(testVecFFN.networkWeights[i], testNonVecFFN.networkWeights[i])

    @given(out1=arrays('float64', (4,), elements=floats(-1,1, width=16), unique=True),
           out2=arrays('float64', (5,), elements=floats(-1,1, width=16), unique=True),
           out3=arrays('float64', (3,), elements=floats(-1,1, width=16), unique=True),
           delta1=arrays('float64', (4,), elements=floats(-1,1, width=16), unique=True),
           delta2=arrays('float64', (3,), elements=floats(-1,1, width=16), unique=True))
    def testForwardWeightUpdateNonVecEqualsVecVersion(self, out1, out2, out3, delta1, delta2):
        testRandom = TestSeqRandom()
        testVecFFN = FeedForwardNetwork(testRandom, [3,4,3], 0.5, 0.6)
        testNonVecFFN = FeedForwardNetwork(testRandom, [3,4,3], 0.5, 0.6)
        
        prevWeights = copy.deepcopy(testVecFFN.networkWeights)
        testVecFFN.networkDeltas = [delta1, delta2]
        testNonVecFFN.networkDeltas = [delta1, delta2]
        outputVectors = [out1, out2, out3]

        testNonVecFFN._forwardWeightsUpdateNonVec(outputVectors)
        testVecFFN._forwardWeightsUpdateVec(outputVectors)

        for i in range(len(testVecFFN.networkWeights)):
            assert_array_equal(testVecFFN.networkWeights[i], testNonVecFFN.networkWeights[i])
            assert_raises(AssertionError, assert_array_equal, testVecFFN.networkWeights[i], prevWeights[i])

    def testCorrectLayerDimentions(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        self.assertEqual(len(TEST_NETWORK_TOPO)-1, len(testFFN.networkWeights))

        for id,layer in enumerate(testFFN.networkWeights):
            (noWeights,noNeurons) = layer.shape
            self.assertEqual(noWeights,TEST_NETWORK_TOPO[id] + 1)
            self.assertEqual(noNeurons,TEST_NETWORK_TOPO[id+1])

    def testCorrectDeltaDimentions(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        self.assertEqual(len(TEST_NETWORK_TOPO)-1, len(testFFN.networkDeltas))

        for id,layer in enumerate(testFFN.networkDeltas):
            self.assertEqual(1, layer.ndim)
            self.assertEqual(layer.size, TEST_NETWORK_TOPO[id+1])

    def testCorrectNormalisation(self):
        vec = np.array([150, 150, 150], dtype='float64')
        actVec = FeedForwardNetwork._normaliseVector(vec, (np.array([50,0, 50], dtype='float64'), np.array([450, 400, 450], dtype='float64')))
        expVec = np.array([0.25, 0.375, 0.25], dtype='float64')
        self.assertTrue(np.array_equal(actVec, expVec))

    def testRaisesInvalidRandomGenerator(self):
        invalidRandom = list()
        self.assertRaises(TypeError, FeedForwardNetwork, invalidRandom, TEST_NETWORK_TOPO, 0, 0)

    def testRaisesInvalidTopology(self):
        invalidTopo = 5
        testRandom = TestConstRandom(1)
        self.assertRaisesRegex(Exception, "Invalid network topology given", FeedForwardNetwork, testRandom, invalidTopo, 0, 0)

    def testCorrectInitializedWeights(self):
        testRandom = TestSeqRandom()
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)
        for layer in testFFN.networkWeights:
            expect = 1
            for x in layer:
                for y in x:
                    self.assertEqual(expect, y)
                    expect += 1

    def testCorrectDataSetNormalisation(self):
        testFFN = Mock(wraps=FeedForwardNetwork, autospec=True)
        testFFN.networkWeights = MagicMock()
        testFFN.networkWeights = [Mock(), Mock()]
        testFFN.networkWeights[0].shape =  [4]
        testFFN.networkWeights[-1].shape = [0,2]
        testFFN.verboseFlag = False
        
        testFFN._testInputVecFormat = lambda x:x
        testFFN._testOutputVecFormat = lambda x:x

        inVec = [np.array([1, 1, 2], dtype='float64'),
                 np.array([6, 1, 1], dtype='float64'),
                 np.array([4, 2, 1], dtype='float64'),
                 np.array([6, 1, 0], dtype='float64')]

        outVec = [np.array([2, 4], dtype='float64'),
                  np.array([2, 3], dtype='float64'),
                  np.array([2, 2], dtype='float64'),
                  np.array([3, 1], dtype='float64')]


        testDataSet = [(inVec[c], outVec[c]) for c in range(len(inVec))]
        
        testFFN._normaliseDataSet(testFFN, testDataSet)

    def testAcceptsCorrectInputVector(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        try:
            inputVector = np.array([1] * TEST_NETWORK_TOPO[0], dtype='float64')
            testFFN._testInputVecFormat(inputVector)

        except Exception:
            self.fail("Raised exception although input is valid")

    def testRaisesInvalidInputVector(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        inputVector = np.array([1] * (TEST_NETWORK_TOPO[0] + 1), dtype='float64')
        self.assertRaisesRegex(Exception, "Invalid input vector format", testFFN._testInputVecFormat, inputVector)

        inputVector = 5
        self.assertRaisesRegex(Exception, "Invalid input vector format", testFFN._testInputVecFormat, inputVector)

    def testAcceptsCorrectOutputVector(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)
        try:
            outputVector = np.array([1] * TEST_NETWORK_TOPO[-1], dtype='float64')
            testFFN._testOutputVecFormat(outputVector)

        except Exception:
            self.fail("Raised exception although output is valid")

    def testRaisesInvalidOutputVector(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, TEST_NETWORK_TOPO, 0, 0)

        outVector = np.array([1] * (TEST_NETWORK_TOPO[-1] + 1), dtype='float64')
        self.assertRaisesRegex(Exception, "Invalid Output vector format", testFFN._testOutputVecFormat, outVector)

        outVector = 5
        self.assertRaisesRegex(Exception, "Invalid Output vector format", testFFN._testOutputVecFormat, outVector)

    def testCorrectLayerOutputCalculation(self):
       
        inputVector = np.array([1, 1], dtype='float64')
        layerWeights = np.ones((3, 2), dtype='float64')

        (newInput, output) = FeedForwardNetwork._calcNextLayerOutput(inputVector, layerWeights)
       
        expectInput = sigmoid(np.array([3, 3], dtype='float64'))
        
        self.assertTrue(np.array_equal(expectInput, output))
        
        self.assertTrue(np.array_equal(newInput, np.array([1] * 3, dtype='float64')))

    def testCorrectForwardPass(self):
        tests = [([3,3,3], [13,13,13]),
                 ([4,1], [5])]

        for (layout, expOut) in tests:
            with self.subTest(test=(layout, expOut)):
                testRandom = TestConstRandom(1)
                testFFN = FeedForwardNetwork(testRandom, layout, 0, 0)

                inputVec = np.ones(layout[0], dtype='float64')
                with patch('ffnBp.ffn.sigmoid', lambda x: x) as _:
                    outputs = testFFN._forwardPass(inputVec)

                    expArray = np.array(expOut, dtype='float64')

                    self.assertTrue(np.array_equal(expArray, outputs[-1]))

    def testCorrectDerivErrorSigmoidHiddenNonVec(self):
        tests = [([2,2,2], [[1,1,1]], 0, 6),
                 ([3,3,3],[[0,0,0],[2,2,2]], 1, 18)]

        for (deltas,weights, curNode, expOut) in tests:
            with self.subTest(test=(deltas,weights,expOut)):
                nextLayerDeltas = np.array(deltas, dtype='float64')
                nextLayerWeights= np.array(weights, dtype='float64')

                result = FeedForwardNetwork._calcDerivErrorSigmoidHiddenNonVec(curNode, nextLayerWeights, nextLayerDeltas)

                self.assertEqual(result, expOut)

    def testCorrectBackwardPassNonVec(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, [3,3,3], learningRate = 0.5, momentum = 0)
        
        expOutVector = MagicMock(spec=np.ndarray)

        outputVectors = [np.array([2,2,2,2]),
                         np.array([2,2,2,2]),
                         np.array([2,2,2])]

        with patch('ffnBp.ffn.FeedForwardNetwork._calcDerivErrorSigmoidOutput', lambda _,__: 1):
            with patch('ffnBp.ffn.FeedForwardNetwork._calcDerivErrorSigmoidHiddenNonVec', lambda _,__,___: 2):
                with patch('ffnBp.ffn.FeedForwardNetwork._forwardWeightsUpdateNonVec', lambda _,__:_):
                    testFFN._backwardPassNonVec(outputVectors, expOutVector)
                    for layerId in range(len(testFFN.networkDeltas)):
                        if layerId == len(testFFN.networkDeltas)-1:
                            self.assertTrue(np.alltrue(testFFN.networkDeltas[layerId] == -2))
                        else:
                            self.assertTrue(np.alltrue(testFFN.networkDeltas[layerId] == -4))

    def testCorrectForwardWeightsUpdateNonVec(self):
        testRandom = TestConstRandom(1)
        testFFN = FeedForwardNetwork(testRandom, [3,3,3], learningRate = 0.5, momentum = 0)

        outputVectors = MagicMock(spec=list)

        with patch('ffnBp.ffn.FeedForwardNetwork._calcDerivErrorWeightNonVec', lambda _,__: 1):
            testFFN._forwardWeightsUpdateNonVec(outputVectors)
            for layerId in range(len(testFFN.networkWeights)):
                self.assertTrue(np.alltrue(testFFN.networkWeights[layerId] == 0.5))
                self.assertTrue(np.alltrue(testFFN.tMinus1NetworkWeights[layerId] == 1))

    def testCorrectErrorRate(self):
        tests = [([1,0,2], [1,0,0], 2),
                 ([1,0,1], [1,0,1], 0),
                 ([1,1], [0,0], 1)
                ]
        
        for (actual,target, expErr) in tests:
            with self.subTest(test= (actual,target, expErr)):
                out1 = np.array(actual, dtype='float64')
                out2 = np.array(target, dtype='float64')

                err = FeedForwardNetwork._calcErrorRate(out1, out2)
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

                out = FeedForwardNetwork._calcOutcomesRoundEach(out1, out2)
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

                out = FeedForwardNetwork._calcOutcomesWinnerTakeAll(out1, out2)
                self.assertEqual(out, expOutcome)

    def testCorrectOverfitDetection(self):
        mockFFN = Mock(wraps=FeedForwardNetwork)
        mockFFN._overfitConseq = OVERFIT_THRESH - 1
        mockFFN._bestTestOutcome = OVERFIT_START + 1
        mockFFN._inStoreWeights = Mock()
        mockFFN._inStoreTMinus1NetworkWeights = Mock()
        mockFFN._inStoreDeltas = Mock()
        mockFFN._inStoreEpoch = Mock()
        self.assertTrue(mockFFN._overfitCheck(mockFFN, OVERFIT_START - 1))
        self.assertEqual(mockFFN._overfitConseq, OVERFIT_THRESH)

    def testCorrectOverfitBreak(self):
        mockFFN = Mock(wraps=FeedForwardNetwork)
        mockFFN._overfitConseq = 30
        mockFFN._bestTestOutcome = OVERFIT_START + 1
        mockFFN.networkWeights = Mock()
        mockFFN.tMinus1NetworkWeights = Mock()
        mockFFN.networkDeltas = Mock()
        mockFFN.epoch = Mock()
        self.assertFalse(mockFFN._overfitCheck(mockFFN, OVERFIT_START + 2))
        self.assertEqual(mockFFN._overfitConseq, 0)
        self.assertEqual(mockFFN._bestTestOutcome, OVERFIT_START + 2)

    def testCorrectEpochReverting(self):
        mockFFN = Mock(wraps=FeedForwardNetwork)
        mockFFN._overfitConseq = 30
        mockFFN._inStoreWeights = Mock()
        mockFFN._inStoreTMinus1NetworkWeights = Mock()
        mockFFN._inStoreDeltas = Mock()
        mockFFN._inStoreEpoch = Mock()
        mockFFN.revert2BestEpoch(mockFFN)
        self.assertEqual(mockFFN._inStoreWeights, mockFFN.networkWeights)

    def testLearningRateDecay(self):
        testRandom = TestSeqRandom()
        testFFN = FeedForwardNetwork(testRandom, 
                                     neuronsPerLayer=[3,4,3], 
                                     learningRate=0.5, 
                                     momentum=0.6,
                                     decayLearningRateFlag=True)

        testFFN.epoch = 90

        adjustedRate = testFFN.learningRate
        self.assertEqual(adjustedRate, gaussianDecay(0.5,90) )