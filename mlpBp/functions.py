import signal
import functools
import math

def sigintHandler(ffnObj, signum, frame):
   # ignore additional SIG_INT signals
   signal.signal(signal.SIGINT, signal.SIG_IGN)

   if input(" Exit training? (Y|n) ") == "Y":
      print("Exiting, please wait..")
      ffnObj._exitFlag = True
   else:
      # re-enable SIG_INT signals
      signal.signal(signal.SIGINT, functools.partial(sigintHandler, ffnObj))
   
def sigmoid(x):
   return 1/(1+math.e**(-x))

def gaussianDecay(start, time):
   return start*math.e**(-time/1000)

