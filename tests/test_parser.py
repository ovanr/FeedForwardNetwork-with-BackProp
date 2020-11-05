import unittest
import numpy as np

import ffnBp.parser as p
from unittest.mock import Mock, patch, MagicMock, mock_open

# to run this test do
# cd 'ffnBp' and run: python -m unittest tests.test_parser -v

class TestFFNParser(unittest.TestCase):
   def testReadDataSet(self):
      rawDat = "0 1 2\n3 4 5\n6 7 8\n\n"
      with patch("builtins.open", mock_open(read_data=rawDat)):
         out = p.readDataSet("not used")
         self.assertEqual(out, [['0','1','2'],['3','4','5'],['6','7','8']])

   def testParseDataSet(self):
      testIn = [['0','1','2'],['3','4','5'],['6','7','8']]

      out = p.parseDataSet(testIn, 1, 2)

      expOut = [(np.array(['0'], dtype='float32'), np.array(['1','2'], dtype='float32')),
                (np.array(['3'], dtype='float32'), np.array(['4','5'], dtype='float32')),
                (np.array(['6'], dtype='float32'), np.array(['7','8'], dtype='float32'))]

      self.assertEqual(len(expOut), len(out))
      for i in range(len(expOut)):
         self.assertTrue(np.array_equal(out[i][0], expOut[i][0]))
         self.assertTrue(np.array_equal(out[i][1], expOut[i][1]))

   def testSplitDataSetCorrectRatio(self):
      a = np.array(['0','1'], dtype='float32')
      b = np.array(['1','0'], dtype='float32')

      testIn = [(np.array(['0'], dtype='float32'), a),
                (np.array(['1'], dtype='float32'), b),
                (np.array(['2'], dtype='float32'), a),
                (np.array(['3'], dtype='float32'), b),
                (np.array(['4'], dtype='float32'), a),
                (np.array(['5'], dtype='float32'), b)]

      (train, test) = p.splitDataSet(testIn)

      def countEachGroup(dataSet):
         groupA = 0
         groupB = 0
         for (_,o) in dataSet:
            if np.array_equal(o,a):
               groupA += 1
            elif np.array_equal(o,b):
               groupB += 1

         return (groupA, groupB)

      (groupA, groupB) = countEachGroup(train)
      self.assertEqual(groupA, 2)
      self.assertEqual(groupB, 2)

      (groupA, groupB) = countEachGroup(test)
      self.assertEqual(groupA, 1)
      self.assertEqual(groupB, 1)

   def testSplitDataSetCorrectRecords(self):
      a = np.array(['0','1'], dtype='float32')
      b = np.array(['1','0'], dtype='float32')

      testIn = [(np.array(['0'], dtype='float32'), a),
                (np.array(['1'], dtype='float32'), b),
                (np.array(['2'], dtype='float32'), a),
                (np.array(['3'], dtype='float32'), b),
                (np.array(['4'], dtype='float32'), a),
                (np.array(['5'], dtype='float32'), b)]

      (train, test) = p.splitDataSet(testIn)
      dataSet = train + test

      seen = {}
      for (i,_) in dataSet:
         self.assertGreaterEqual(i[0], 0)
         self.assertLessEqual(i[0], 5)
         self.assertNotIn(i[0], seen)
         seen[i[0]] = True

   def testCorrectReadParams(self):
      rawDat = "name1 val1\nname2 val2\n\n"
      with patch("builtins.open", mock_open(read_data=rawDat)):
         out = p.readParams("not used")
         self.assertEqual(out, {'name1': 'val1', 'name2': 'val2'})

   def testRemoveBlankLinesAndComments(self):
      rawDat = "name1 val1\n" + \
               "# this is a comment\n" + \
               "\n" + \
               "name2 val2"

      with patch("builtins.open", mock_open(read_data=rawDat)):
         out = p.readParams("not used")
         self.assertEqual(out, {'name1': 'val1', 'name2': 'val2'})

   def testGetLayout(self):
      params = {
         'numInputNeurons': 1,
         'numHiddenLayerOneNeurons': 2,
         'numHiddenLayerTwoNeurons': 3,
         'numHiddenLayerThreeNeurons': 0,
         'numHiddenLayerFourNeurons': 5,
         'numOutputNeurons': 6
      }

      layout = p.getLayout(params)
      self.assertEqual(layout, [1,2,3,5,6])

   def testGetOutcomeCalcMethod(self):
      tests = [ { 'outcomeCalcMethod': 'winner-take-all' },
                { 'outcomeCalcMethod': 'round-each' }]

      for dic in tests:
         with self.subTest(test=dic):
            ret = p.getOutcomeCalcMethod(dic)
            self.assertIsNotNone(ret)

      with self.subTest(test={}):
         ret = p.getOutcomeCalcMethod({})
         self.assertIsNone(ret)

   def testParseMissingParams(self):
      params = {
         'dataFile'      : '_',
         'numInputNeurons' : 3,
         'numOutputNeurons' : 3
      }
      
      expOut = {
         "learningRate" : 0.5,
         "decayLearningRate" : False,
         "momentum"     : 0.6,
         "maxIterations": 2000,
         "minErrorRate" : 0.0001,
         "verbose"      : False,
         'numInputNeurons' : 3,
         'numOutputNeurons' : 3,
         'layout'       : [],
         'outcomeCalcMethod': None,
         'trainSet'     : 1,
         'testSet'      : 2
      }

      with patch('ffnBp.parser.getOutcomeCalcMethod', lambda x: None):
         with patch('ffnBp.parser.getLayout', lambda x: []):
            with patch('ffnBp.parser.readDataSet', lambda x:x):
               with patch('ffnBp.parser.parseDataSet', lambda x,__,___:x):
                  with patch('ffnBp.parser.splitDataSet', lambda x:(1,2)):
                     out = p.parseParams(params)
                     self.assertEqual(out, expOut)

   def testParseParams(self):
      params = {
         "learningRate" : '0.45',
         "decayLearningRate": "True",
         "momentum"     : '8.55',
         "maxIterations": '1001',
         "minErrorRate" : '0.01',
         "verbose"      : 'True',
         'dataFile'      : '_',
         'numInputNeurons' : 3,
         'numOutputNeurons' : 3
      }
      
      expOut = {
         "learningRate" : 0.45,
         "decayLearningRate": True,
         "momentum"     : 8.55,
         "maxIterations": 1001,
         "minErrorRate" : 0.01,
         "verbose"      : True,
         'numInputNeurons' : 3,
         'numOutputNeurons' : 3,
         'layout'       : [],
         'outcomeCalcMethod': None,
         'trainSet'     : 1,
         'testSet'      : 2
      }

      with patch('ffnBp.parser.getOutcomeCalcMethod', lambda x: None):
         with patch('ffnBp.parser.getLayout', lambda x: []):
            with patch('ffnBp.parser.readDataSet', lambda x:x):
               with patch('ffnBp.parser.parseDataSet', lambda x,__,___:x):
                  with patch('ffnBp.parser.splitDataSet', lambda x:(1,2)):
                     out = p.parseParams(params)
                     self.assertEqual(out, expOut)
