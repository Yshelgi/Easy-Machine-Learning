#coding=utf-8
import feedparser
import re
from numpy import *


#包含所有文档 不含重复词的list
def createVocabList(dataSet):
    vocabSet=set([])#创建空集，set是返回不带重复词的list
    for document in dataSet:
        vocabSet=vocabSet|set(document) #创建两个集合的并集
    return list(vocabSet)
#判断某个词条在文档中是否出现
def setOfWords2Vec(vocabList, inputSet):#参数为词汇表和某个文档
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词: %s 不在我的词汇里面!" % word)#返回文档向量 表示某个词是否在输入文档中出现过 1/0
    return returnVec


#朴素贝叶斯分类训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):#遍历每个文档
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

#给定词向量 判断类别
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


#示例：过滤垃圾邮件
#预处理
def textParse(bigString):
    import re
    listOfTokens=re.split('x*',bigString)  #接收一个大字符串并将其解析为字符串列表
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #去掉少于两个的字符串并全部转化为小写
#过滤邮件 训练+测试
def spamTest():
    docList=[]; classList=[]; fullText=[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet = list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
            print("分类错误的是： %s" %docList[docIndex])
    print('错误率是:',float(errorCount)/len(testSet))

# spamTest()
spamTest()
