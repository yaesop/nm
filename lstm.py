# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import os
from numpy import loadtxt
import numpy as np
import os
import keras

# Binary Classification with Sonar Dataset: Baseline
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import StratifiedKFold
import time

# load the dataset

windowSize = 10
data = [(1,4,1,4),(1,4,4,1), (1,5,4,1),(1,7,1,1),(1,8,3,1),(1,9,4,1),(1,11,2,4),(2,5,4,1),(2,5,4,4),(2,7,1,4),(2,8,2,1),(2,8,3,1),(2,8,4,1),(2,9,4,1),(2,12,3,1)] #cell#, mouseNumber, day, session 

for pair in data:
    if pair[1]< 10:
        cellMouse = str(pair[0]) + "00" + str(pair[1]) 
    else:
        cellMouse = str(pair[0]) + "00" + str(pair[1]) 
    inj = pair[3]
    day = pair[2]

    datasetName = os.path.join('../data',cellMouse,'TRACES_'+ cellMouse +'_'+ str(day) +'_' +str(inj) +'.csv')
    labelName = os.path.join('../data',cellMouse,'BEHAVIOR_'+ cellMouse +'_'+ str(day) +'_' +str(inj)+'.csv')
    inj +=1 
    datasetName1 = os.path.join('../data',cellMouse,'TRACES_'+ cellMouse +'_'+ str(day) +'_' +str(inj) +'.csv')
    labelName1 = os.path.join('../data',cellMouse,'BEHAVIOR_'+ cellMouse +'_'+ str(day) +'_' +str(inj)+'.csv')

    inj += 1
    testName = os.path.join('../data',cellMouse,'TRACES_'+ cellMouse +'_'+ str(day) +'_' +str(inj) +'.csv')
    testLabel = os.path.join('../data',cellMouse,'BEHAVIOR_'+ cellMouse +'_'+ str(day) +'_' +str(inj)+'.csv')

    dataset = loadtxt(datasetName, delimiter=',')
    label = loadtxt(labelName, delimiter=',')
    dataset1 = loadtxt(datasetName1, delimiter=',')
    label1 = loadtxt(labelName1, delimiter=',')
    testDataset = loadtxt(testName, delimiter=',')
    testLabel = loadtxt(testLabel, delimiter=',')


    max_trainval = [] 
    min_trainval = [] 
    mean1_trainval = [] 
    mean2_trainval = [] 
    mean3_trainval = [] 
    std1=[]
    std2=[]
    std3=[]
    
    Nfeature = dataset.shape[0] 

   
        
    
    #print(max_trainval,min_trainval ,mean_trainval)

    X = np.transpose(dataset)
    Y = label[:,2]

    X2 = np.transpose(dataset1)
    Y2 = label1[:,2]

    testX = np.transpose(testDataset)
    testY = testLabel[:,2]

    for i in range(3000):
        X[i][0] = 0
        X2[i][0] = 1
        testX[i][0] = 2
    
    paddedX = X
    paddedX2 = X2
    paddedtestX = testX
    #print(np.shape(paddedX2), X)
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    encoder.fit(Y2)
    encoded_Y2 = encoder.transform(Y2)
    encoder.fit(testY)
    encoded_testY = encoder.transform(testY) 

    
    model = Sequential()
    model.add(keras.layers.LSTM(100, input_shape=(windowSize, X.shape[1]),return_sequences=False))

    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    

    
    for ep in range(30):
        for i in range(windowSize,3000):
            model.fit(paddedX[i-windowSize:i][:].reshape(1,windowSize,X.shape[1]),encoded_Y[i].reshape(1,1),epochs=1, batch_size=1,verbose=0) 
        for i in range(windowSize,3000):
            model.fit(paddedX2[i-windowSize:i][:].reshape(1,windowSize,X2.shape[1]),encoded_Y2[i].reshape(1,1),epochs=1, batch_size=1,verbose=0) 
       #print(ep,"/30")

    #Test#
    results=[]


    max_time = 0
    for i in range(windowSize,3000):
        start_time = time.time()
        results.extend(model.predict(paddedtestX[i-windowSize:i][:].reshape(1,windowSize,testX.shape[1]))[0]) 
        model.fit(paddedtestX[i-windowSize:i][:].reshape(1,windowSize,paddedtestX.shape[1]), \
        encoded_testY[i].reshape(1,1),epochs=1, batch_size=1,verbose=0)
        temp_time = time.time() - start_time
        if max_time < temp_time:
            max_time = temp_time
    #for i in range(2000+windowSize,3000):
    #    results.extend(model.predict(paddedtestX[i-windowSize:i][:].reshape(1,windowSize,testX.shape[1]))[0])
 
    print("--- %s seconds ---" % (max_time))

    np.reshape(results, (3000-windowSize))
    
    accuracy = 0 
    fn = 0
    fp = 0
    tp = 0 
    for i in range(3000-windowSize):
        if results[i]> 0.5:
            results[i]= 1 
        else:
            results[i] = 0 
        if results[i]==testY[i+windowSize] and results[i]==1:
            accuracy +=1
            tp +=1
        elif results[i]==testY[i+windowSize] and results[i]==0:
            accuracy +=1
        elif results[i]!=testY[i+windowSize] and results[i]==1:
            fp += 1
        else:
            fn += 1

    #print("balance:",sum(results), sum(testY))
    print(datasetName)
    print(datasetName1)
    print(testName)
    print(accuracy*100/(3000-windowSize))
    print("f1_score:" ,tp/(tp+(fn+fp)/2) )


