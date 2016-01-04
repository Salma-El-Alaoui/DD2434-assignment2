__author__ = 'Salma'

import numpy as np
import matplotlib.pyplot as plt

fileName = "dataICA.txt"
# observed data over time (x1, x2) in (N*2)
X = np.loadtxt(fileName,skiprows=2)

#shift the column-wise mean to 0
def center(X):
    centered =  X - np.mean(X, axis =0)
    return np.transpose(centered)

def whiten(X):
    Y = np.transpose(X)
    N, p = Y.shape
    Y= Y/np.sqrt(N-1)
    U, s, V = np.linalg.svd(Y, full_matrices=False)
    S= np.diag(s)
    #np.allclose(X, np.dot(U, np.dot(S, V)))
    Yt = np.transpose(Y)
    YtY = np.dot(Yt, Y)
    #diagonal matrix of eigen values of XXt:
    D = S**2
    #orthogonal matrix of eigen vectors of XXt:
    E = np.transpose(V)
    ED = np.dot(E,D)
    #print(np.dot(E, np.transpose(E)))
    #print(np.allclose(XXt, np.dot(ED, np.transpose(E))))
    R = np.dot(E, np.dot(np.linalg.inv(S), np.transpose(E)))
    #Whitened mixture
    Xtilda = np.dot(R,X)
    #print(np.diag(np.dot(Xtilda,np.transpose(Xtilda))))
    return Xtilda, E, D

def fastICA(X):
    p, N = X.shape
    W = np.zeros((p,p))
    iterations = 10
    #number of componenets
    for i in range(p):
        #random initialisation
        W[i,:] = np.sqrt(1) * np.random.randn(p)
        for k in range(iterations):
            wtold = np.transpose(W[i,:])
            g = np.tanh(np.dot(wtold,X))
            gPrime = np.ones((1,N))- np.multiply(np.tanh(np.dot(wtold,X)), np.tanh(np.dot(wtold,X)))
            w = 1/N*np.dot(X, np.transpose(g))- np.mean(gPrime)*W[i,:]
            w = w/np.sqrt(np.dot(np.transpose(w),w))
            if i == 1:
                w = w - np.dot(W[0,:], np.dot(np.transpose(w),W[0,:]))
                w = w/np.sqrt(np.dot(np.transpose(w),w))
            #check convergence:
            #if np.allclose(1, np.dot(W[i,:],w)):
                #print(np.dot(W[i,:],w))
                #W[i,:] = w
            #print("iteration",k, "  ",np.dot(W[i,:],w))
            W[i,:] = w
    S = np.dot(W,X)
    A = np.linalg.inv(W)
    return W,S,A


Xcenter = center(X)
Xwhite, E, D= whiten(Xcenter)
W,S,A = fastICA(Xwhite)

#whitened X
plt.scatter(Xwhite[0,:],Xwhite[1,:])
#plt.plot(Xwhite[0,:], Xwhite[1,:], marker='.', color='blue')
plt.xlabel(r"$\widetilde{x}_1$")
plt.ylabel(r"$\widetilde{x}_2$")
ax = plt.axes()
ax.arrow(0, 0, A[0,1],A[1,1], head_width=0.06, head_length=0.1, fc='k', ec='k',  linewidth=2.0)
ax.arrow(0, 0, A[0,0],A[1,0], head_width=0.06, head_length=0.1, fc='k', ec='k',  linewidth=2.0)
plt.show()

#centered data + eigen vectors +eigen values
plt.scatter(Xcenter[0,:],Xcenter[1,:], alpha ='0.5')
D = np.sqrt(np.diag(D))
ax = plt.axes()
ax.arrow(0, 0, E[0,0]*D[0],E[1,0]*D[0], head_width=0.06, head_length=0.1, fc='r', ec='r',  linewidth=2.0)
ax.annotate('ev1 = 5.21', xy=(E[0,0]*D[0],E[1,0]*D[0] ), xytext=(E[0,0]*D[0]+0.4, E[1,0]*D[0]+0.7), fontweight='bold',color='r')

ax.arrow(0, 0, E[0,1]*D[1],E[1,1]*D[1], head_width=0.06, head_length=0.1, fc='r', ec='r',  linewidth=2.0)
ax.annotate('ev2 = 0.05', xy=(E[0,1]*D[1],E[1,1]*D[1]), xytext=(E[0,1]*D[1]+0.15, E[1,1]*D[1]+0.15), fontweight='bold', color ='r')

#plt.plot([0, E[0,0]*D[0]],[0, E[1,0]*D[0]],'g', linewidth=2.0)
#plt.plot([0,E[0,1]*D[1]],[0, E[1,1]*D[1]],'r', linewidth=2.0)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()

# 2D dstribution Signals
plt.scatter(S[0,:],S[1,:])
plt.show()
time = np.linspace(0,100,5000)
plt.plot(time,S[0,:] )
plt.show()
plt.plot(time,S[1,:] )
plt.show()






