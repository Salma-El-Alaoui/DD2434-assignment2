__author__ = 'Salma'

import numpy as np
import matplotlib.pyplot as plt

#generates a sequence of length K of the variable Z (a sequence of hidden states)
# Z = 1 if the table is primed and 0 if the table is not primed
def generateTables(K, initProb):
    sequence =[]
    #sample first table
    s = np.random.binomial(1, initProb, 1)
    #sample k-1 tables
    sequence.append(s[0])
    for k in range(K-1):
        if sequence[len(sequence)-1] == 0:
            s = np.random.binomial(1,3/4,1)
        else:
            s = np.random.binomial(1,1/4,1)
        sequence.append(s[0])

    return sequence

#Samples from the outcome of the table's dice
#primed and unPrimed are the categorical distributions for K primed and K unprimed Tables
def sampleTableDice(primed, unPrimed, tables):
    sequence = []
    for k in range(len(tables)):
        if tables[k] == 0:
            s = np.random.multinomial(1, unPrimed[k, :])
        else:
            s = np.random.multinomial(1, primed[k, :])
        outcome = np.argmax(s)+ 1
        sequence.append(outcome)
    return(sequence)

#Samples from the outcome of the player's dice
def samplePlayerDice(playerDice, tables):
    sequence = []
    for k in range(len(tables)):
        s =  np.random.multinomial(1, playerDice)
        outcome = np.argmax(s)+ 1
        sequence.append(outcome)
    return(sequence)

def demo(nTables, primed, unPrimed, playerDice):

    tables = generateTables(nTables, 0.5)
    tableOutcome = np.asarray(sampleTableDice(primed, unPrimed, tables))
    playerOutcome = np.asarray(samplePlayerDice(playerDice, tables))
    sum = tableOutcome + playerOutcome
    return sum

#compute an element of the B matrix ie:
#p(Sk = observation|Zk = state, model)
#time step starts at 0
def computeB(observation, k, state, primed, unPrimed, playerDice):
    #we first find out what the table distribution is given the state
    if state == 0:
        tableDice = unPrimed[k,:]
    else :
        tableDice = primed[k,:]
    #let's compute the probability:
    prob = 0
    for i in range(len(playerDice)):
        for j in range(len(tableDice)):
            if (i+j+2 == observation):
                prob += playerDice[i]*tableDice[j]
    return prob


# compute alpha given a sequence of observations and the parameters of the model
# primed: distribution of the dice in the primed tables
# unprimed: distribution of the dice in the unprimed tables
# playerDice : distribution of the player's dice
# A: transition matrix (for the states)
# pi: intial state distribution
def SampleFromPosterior(obs, primed, unPrimed, playerDice, pi, A):
    #number of states
    states = np.size(A[0,:])
    observations = len(obs)
    #table of alphas
    alpha = np.zeros((observations,states))
    #alpha(timestep, state)
    for i in range(states):
        alpha[0][i] = pi[i]*computeB(obs[0],0,i,primed, unPrimed,playerDice)

    for k in range(1,observations):
        for i in range(states):
            for j in range(states):
                alpha[k][i] += alpha[k-1][j]*A[j][i]*computeB(obs[k],k,i, primed, unPrimed, playerDice)
    norm = 0
    for i in range(states):
        norm += alpha[observations-1][i]
    prob1 = alpha[observations-1][1]/norm
    #prob0 = alpha[observations-1][0]/norm
    #print("prob 0",prob0)
    #print("prob 1", prob1)
    zk = np.random.binomial(1,prob1)
    stateSequence = []
    for i in reversed(range(observations-1)):
        zkPrev = []
        for previous in range(states):
            zkPrev.append(A[previous][zk]*alpha[i][previous])
        factor = 0
        for i in range(states):
            factor += zkPrev[i]
        probZkPrev = zkPrev[1]/factor
        stateSequence.append(np.random.binomial(1,probZkPrev))

    #in the order of the observations
    stateSequence = stateSequence[::-1]
    stateSequence.append(zk)
    return stateSequence


def computeTransitionMatrix():
    A = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            if i == j:
                A[i][j]= 3/4
            else :
                A[i][j] = 1/4
    return A

nTables = 10
#initial state distribution
pi = [0.5, 0.5]
tables = generateTables(nTables, pi[1])
print("tables", tables)
#unbiased dice
primed = np.ones((nTables,6))*1/6
playerDice = np.ones(6)*1/6
A = computeTransitionMatrix()
#biased dice for all unprimed tables
unPrimed = np.zeros((nTables,6))
unPrimed[:,0] = 0.5
unPrimed[:,1] = 0.5
obs = demo(nTables, primed, unPrimed, playerDice)
#playerDice = np.zeros(6)
#playerDice[4] = 0.5
#playerDice[5] = 0.5
'''
iter = 10000
hist = demo(nTables, primed, unPrimed, playerDice)
print(hist)
for i in range(iter):
    obs = demo(nTables, primed, unPrimed, playerDice)
    hist = np.append(hist, obs)
n, bins, patches = plt.hist(hist)
plt.xlim(2,12)
plt.show()
'''
print("Observations", obs)
print("Posterior",SampleFromPosterior(obs, primed, unPrimed, playerDice, pi, A))
