"""
 number of consequtive epochs that produced
 test correct outcome percentage lower than 
 the best percentage observed so far. Once 
 this limit is reached, the network is 
 reverted back to the epoch that produced
 the best test outcome percentage and training 
 stops.
"""
OVERFIT_THRESH = 700
OVERFIT_START = 86

DECAY_RATE = 0.3