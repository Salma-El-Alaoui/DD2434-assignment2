import numpy as np
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix

#number of topics

# observed data over time (x1, x2) in (N*2)

topics = 10
alpha = 0.3
beta = 0.3
dic = "R3_all_Dictionary.txt"
data = "R3-trn-all_run.txt"
words = sum(1 for line in open(dic))
documents = sum(1 for line in open(data))
wordStrings = []
fDic = open(dic, 'r')
for wordString in fDic:
    wordStrings.append(wordString[:-1])
#create a document-word matrix:
matrix = np.zeros((documents, words))
f = open(data,'r')
for document,line in enumerate(f):
    for i,word in enumerate(line.split()):
        if i != 0:
            wordCount = word.split(':')
            wordID = int(wordCount[0])
            count = int(wordCount[1])
            matrix[document,wordID] = count

#number of instances of topic z and document m
nmz = np.zeros((documents, topics))
#number of instances of w assigned to topic z
nzw = np.zeros((topics, words))
nm = np.zeros(documents)
nz = np.zeros(topics)
topicAssg = {}
for m in range(documents):
    for w in range(words):
        #if the word w is in document m
        if(matrix[m,w] != 0):
            #choose and arbitrary topic for word i:
            z = np.random.randint(topics)
            nzw[z,w] += 1
            nmz[m,z] += 1
            nm[m] += 1
            nz[z] += 1
            topicAssg[(m,w)] = z

def conditional_distribution(m,w):
    term1 = (nzw[:,w] + beta)/(nz + beta * words)
    term2 = (nmz[m,:] + alpha) / (nm[m] + alpha * topics)
    pz = term1 * term2
    # normalize to obtain probabilities
    pz /= np.sum(pz)
    return pz

def runSampler(iter = 30):
    for it in range(iter):
        for m in range(documents):
            for w in range(words):
                #if the word w is in document m
                if(matrix[m,w] != 0):
                    z = topicAssg[(m,w)]
                    nzw[z,w] -= 1
                    nmz[m,z] -= 1
                    nm[m] -= 1
                    nz[z] -= 1

                    pz = conditional_distribution(m, w)
                    z =  np.random.multinomial(1,pz).argmax()

                    nzw[z,w] += 1
                    nmz[m,z] += 1
                    nm[m] += 1
                    nz[z] += 1
                    topicAssg[(m,w)] = z

def phi():
    num = nzw + beta
    num /= np.sum(num, axis=1)[:, np.newaxis]
    return num

def theta():
    num = nmz + alpha
    num /= np.sum(num, axis=1)[:, np.newaxis]
    return num

def commonWords(n):
    phiMatrix = phi()
    args = np.argsort(-1*phiMatrix, axis=1)[:,:n]
    commonWords = []
    for topic in range(topics):
        topicList = []
        for j in range(n):
            indexWord = args[topic,j]
            word = wordStrings[indexWord]
            phiVal = phiMatrix[topic,indexWord]
            topicList.append((word, phiVal))
        commonWords.append(topicList)
    return commonWords

def topicDistributionDocument(document):
    wordMatrix = matrix[document,: ]
    #list of words in the document with their occurences
    documentList = []
    for i in range(words):
        count = wordMatrix[i]
        if count != 0:
            z = topicAssg[(document, i)]
            documentList.append((wordStrings[i],count, z))
    thetaMatrix = theta()
    topicDis = thetaMatrix[document,:]
    return documentList,topicDis

def computeThetaTest():
    data = "R3-tst-all_run.txt"
    documents = sum(1 for line in open(data))
    #create a test document-word matrix:
    matrix = np.zeros((documents, words))
    f = open(data,'r')
    for document,line in enumerate(f):
        for i,word in enumerate(line.split()):
            if i != 0:
                wordCount = word.split(':')
                wordID = int(wordCount[0])
                count = int(wordCount[1])
                matrix[document,wordID] = count
    phiMatrix = phi()
    #we compute theta for each document by summing over all the B_k for all the words
    #in that dcocument
    thetaTest = np.zeros((documents, topics))
    for m in range(documents):
        for word in range(words):
            if matrix[m,word]!= 0:
                #normalize beta for this word
                normalizedPhi = matrix[m,word]*phiMatrix[:,word]/sum(phiMatrix[:,word])
                thetaTest[m,:] += normalizedPhi
    for i in range(documents):
        thetaTest[i,:] /= sum(thetaTest[i,:])
    return thetaTest

def computeNewTheta():
    data = "R3-tst-all_run.txt"
    documents = sum(1 for line in open(data))
    #create a test document-word matrix:
    matrix = np.zeros((documents, words))
    f = open(data,'r')
    for document,line in enumerate(f):
        for i,word in enumerate(line.split()):
            if i != 0:
                wordCount = word.split(':')
                wordID = int(wordCount[0])
                count = int(wordCount[1])
                matrix[document,wordID] = count
    for i in range(documents):
        matrix[i,:] /= sum(matrix[i,:])
    phiMatrix = phi()
    t = np.transpose(phiMatrix)
    tt = np.dot(phiMatrix, t)
    print(tt)
    inverse = np.linalg.pinv(tt)
    print(inverse)
    thetaTest = np.dot(matrix, np.dot(t, inverse))
    #for i in range(documents):
    #  thetaTest[i,:] /= sum(thetaTest[i,:])
    return thetaTest

def classifyDocuments(trainingSet, testSet, trainLabels, testLabels, k):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(trainingSet, trainLabels)
    result = clf.predict(testSet)
    accuracy = clf.score(testSet, testLabels)
    conf = confusion_matrix(testLabels, result)
    return result, accuracy, conf

def readLabels(file):
    L = np.loadtxt(file)
    return L



runSampler(1000)
trainLabels = "R3-Label.txt"
testLabels = "R3-GT.txt"
trainingData = theta()
testData = computeThetaTest()
labelsTe = readLabels(testLabels)
labelsTr = readLabels(trainLabels)
result, acc, conf = classifyDocuments(trainingData, testData, labelsTr, labelsTe, 171)
print(acc)
print(conf)


