import numpy as np

from tensorflow.keras.models import load_model

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


inputstring="HHEDDCCCBB99877655544433322222111100000"

inputid = np.zeros([39])

for index, char in enumerate(inputstring):
    inputid[index] = l2id(char)
 
inputid = inputid.reshape ([1,39,1])       
output=""

for i in range (0,5):

    model = load_model("output"+str(i)+".h5")
    
    prediction = model.predict(inputid)
    
    print(prediction)
    print(np.argmax(prediction))