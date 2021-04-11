import pandas as pd
import numpy as np


from tensorflow.keras import Sequential, utils, backend
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Activation

data = pd.read_csv("ns-data.csv")
processed_data = pd.DataFrame(columns = ["Input","Output0","Output1","Output2","Output3","Output4"])

def split(word): 
    return [char for char in word] 

#function to convert letter to id
def l2id(letter):
    if letter == 'A':
        id= 10
    elif letter == 'B':
        id= 11
    elif letter == 'C':
        id= 12
    elif letter == 'D':
        id= 13
    elif letter == 'E':
        id= 14
    elif letter == 'F':
        id= 15
    elif letter == 'G':
        id= 16
    elif letter == 'H':
        id= 17
        
    else:
        id = letter
        
    return id
      


for id in range(0,data.shape[0]):
    
    
    #converting input to numbers
    
    inputstring = split(data.iloc[id,0])
    inputid = np.zeros([39])
    #print (inputstring)

    for index, char in enumerate(inputstring):
        inputid[index] = l2id(char)
    #print(inputid)
    processed_data = processed_data.append({ "Input": inputid},ignore_index=True) #Creating new row
    
    #converting output to numbers
    outputstring = split(data.iloc[id,1])

    #print (outputstring)

    for index, char in enumerate(outputstring):

        if char != '-':
            processed_data.iloc[id, int(index/2) + 1 ] = l2id(char) #changing nan values of that row
            



#converting dataframe to numpy arrays and splitting into train and test data
N_TRAIN = 900
N_INPUTS = 39
N_FEATURES = 1
N_OUTPUTS = 18

    
X_train = processed_data.iloc[:N_TRAIN,0].to_numpy()
X_train = np.concatenate(X_train).reshape(N_TRAIN,N_INPUTS,N_FEATURES)
Y_train = []

X_test = processed_data.iloc[N_TRAIN:,0].to_numpy()
X_test = np.concatenate(X_test).reshape(processed_data.shape[0] - N_TRAIN,N_INPUTS,N_FEATURES)
Y_test = []

#one hot encoding of train and test labels
for i in range (0,5):
    Y_train.append(utils.to_categorical(processed_data.iloc[:N_TRAIN,i+1].to_numpy() , N_OUTPUTS))
    Y_test.append(utils.to_categorical(processed_data.iloc[N_TRAIN:,i+1].to_numpy() , N_OUTPUTS))




#Building the model, training, evaluation and save
N_BLOCKS = 512




for i in range (0,5):
    backend.clear_session()
    model = Sequential()
    model.add(LSTM(N_BLOCKS, input_shape=(N_INPUTS, N_FEATURES)))
    model.add(Dense(N_OUTPUTS,activation ="softmax"))
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training and evaluating character "+str(i)+" in Output")
    history = model.fit(X_train, Y_train[i], epochs=1000, validation_split=0.1, verbose=1, batch_size=3)
    print("Evaluating on test data:")
    test_loss, test_acc = model.evaluate(X_test,  Y_test[i], verbose=1)
    model.save("output"+str(i)+".h5")

