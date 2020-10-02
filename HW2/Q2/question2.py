import numpy as np 
import matplotlib.pyplot as plt

# The gambler aims to win 100 dollars
GAMBLE_GOAL = 100

# The probability of the coin coming up heads is 0.4
# PROB_HEADS = 0.25
PROB_HEADS = 0.25

# Stopping threshold for value iteration
THETA = 1e-5

# Undiscounted Episodes
GAMMA = 1

# Reward for wining the gamble
REWARD = 1

def Actions(s):
    PossibleStakes = np.arange(0,min(s, GAMBLE_GOAL - s)+1)
    return PossibleStakes

states = range(0, GAMBLE_GOAL + 1) # The 0 and GAMBLE_GOAL are dummy states
policy = np.zeros(GAMBLE_GOAL + 1)

def ValueIteration(Values, THETA, SweepNumbers):
    
    # Policy Evaluation Sweeps
    for sweep in range(SweepNumbers):
        UpdatedValues = np.zeros(len(states))
        UpdatedValues[GAMBLE_GOAL] = 1
        Delta = THETA + 10
        while Delta > THETA:
            for s in states[1:GAMBLE_GOAL]:
                PossibleActions = Actions(s)
                ActionsValues = []
                for a in PossibleActions:
                    ActionsValues.append(PROB_HEADS * Values[s + a] + (1 - PROB_HEADS) * Values[s - a])
                ActionIndex = np.argmax(ActionsValues)
                UpdatedValues[s] = ActionsValues[ActionIndex]
            Delta = np.sum(abs(UpdatedValues - Values)) 
            Values[:] = UpdatedValues
            
     # Policy Improvement Sweep
        for s in states[1:GAMBLE_GOAL]:
            PossibleActions = Actions(s)
            ActionsValues = []
            for a in PossibleActions:
                ActionsValues.append(PROB_HEADS * Values[s + a] + (1 - PROB_HEADS) * Values[s - a])
            MaxReturnActionIndex = np.argmax(ActionsValues)
            policy[s] = PossibleActions[MaxReturnActionIndex]
            
    return Values, policy 
                
Values = np.zeros(len(states))    
Values[GAMBLE_GOAL] = 1

SweepNumbers = 1
                
Values, policy = ValueIteration(Values, THETA, SweepNumbers)        

    
plt.figure(1)
plt.subplot(211)
plt.plot(range(0,101),Values)
plt.xticks([1, 25, 50, 75, 99])

plt.subplot(212)
x = np.linspace(0, 100, 101)
plt.xticks([1, 25, 50, 75, 99])
plt.stem(x, policy, '-.', bottom=-2)

plt.savefig('question2.png')