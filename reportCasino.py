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

#Observations: sum of the two dices
def demo(nTables, primed, unPrimed, playerDice):
    tables = generateTables(nTables, 0.5)
    tableOutcome = np.asarray(sampleTableDice(primed, unPrimed, tables))
    playerOutcome = np.asarray(samplePlayerDice(playerDice, tables))
    sum = tableOutcome + playerOutcome
    return sum

nTables = 10
#initial state distribution
pi = [0.5, 0.5]
tables = generateTables(nTables, pi[1])
#unbiased dice
primed = np.ones((nTables,6))*1/6
unPrimed = np.ones((nTables,6))*1/6
playerDice = np.ones(6)*1/6
#observations
observations = demo(nTables, primed, unPrimed, playerDice)
