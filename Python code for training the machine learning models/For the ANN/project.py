
    #Importing the necessary libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

from keras.models import load_model   #for importing the saved model
from sklearn.preprocessing import MinMaxScaler



    #Loading the csv file of the audio dataset to check available records for each class
metadata = pd.read_csv('E:/Users/DENNIS/PycharmProjects/pythonProject/Final-Year-Project/Prepared beehive dataset/metadata.csv')
print(metadata.head())

# Feature extraction function
def features_extractor(file):
    # load the file (audio)
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # we extract mfccs
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # in order to find out scaled features we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


#Now we iterate through every audio file and extract features for each audio file using Mel-Frequency Cepstral Coefficients
audio_dataset_path = 'Prepared beehive dataset'

extracted_features=[]

for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),str(row["folder"])+'/',str(row["filename"]))
    final_class_labels=row["label"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


     #converting extracted_features to Pandas dataframe to save as pickle file and then split the data
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','label'])

#saving the extracted features as a pickle file since this format doesnt distort dimesions of extracted arrays like the csv file on reading
extracted_features_df.to_pickle('E:/Users/DENNIS/PycharmProjects/pythonProject/Final-Year-Project/Prepared beehive dataset/features_normalised.pkl')


    #Loading the created dataframe of the extracted features for use
extracted_features_df = pd.read_pickle('E:/Users/DENNIS/PycharmProjects/pythonProject/Final-Year-Project/Prepared beehive dataset/features_normalised.pkl')

    #Spliting the dataset into independent and dependent dataset and converting them into a numpy array
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['label'].tolist())

    #Label Encoding, involving representing labels as numbers: 0 & 1
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


    #Splitting the data into training data and a remainder from which validation and training sets will be split
X_train,X_rem,y_train,y_rem = train_test_split(X,y,train_size=0.8,random_state=0)

    #It is a good practice to have validation and training data so that testing data is introduced at the end when it is new to the machine
test_size = 0.5
X_valid,X_test,y_valid,y_test = train_test_split(X_rem,y_rem,test_size=0.5,random_state=0)

print(X.shape)
print('\ny',y.shape)


     # Building the Model

# No of labels, which will be 2 in this case since we expect absence or presence of a queen bee
num_labels=y.shape[1]

model=Sequential()
#first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()   #Returns the model architecture

    #Compiling the model
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    #Training the built model
num_epochs = 100
num_batch_size = 32

    #The filepath is where our model is to be saved
checkpointer = ModelCheckpoint(filepath='E:/Users/DENNIS/PycharmProjects/pythonProject/Final-Year-Project/Trained ANN model/FYP_Beehivemodel_normalised.hdf5',
 verbose=1, save_best_only=True)
start = datetime.now()
history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_valid, y_valid),
callbacks=[checkpointer], verbose=1)  #Adding the callbacks to the fit method enables saving the model
duration = datetime.now() - start
print("Training completed in time: ", duration)



   #plotting a graph for training accuracy vs validation accuracy over the number of epochs
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1,101)
plt.plot(epochs, train_acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

    #plotting a graph for training loss vs validation loss over the number of epochs
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


    #loading the saved model
model = load_model('Trained ANN model/FYP_Beehivemodel.hdf5')

#     #Checking if the loaded model has the expected architecture, weights and optimizer
# model.summary()
# print(model.get_weights())
# print(model.optimizer)
#
# #Evaluating the trained model
# test_accuracy=model.evaluate(X_test,y_test,verbose=0)
# print("The test accuracy is: ",test_accuracy[1])


    #Testing with actual audio files
NoQB = 'Testing_data/NoQueenBee'
QBP = 'Testing_data/QueenBee_present'

    #Function for getting files from a passed directory
def getFilesInDir(directory):
    files = []
    files = os.listdir(directory)
    return files

NoQB_files = getFilesInDir(NoQB)  #Array of audio files with No Queen Bee
QBP_files = getFilesInDir(QBP)    #Array of audio files for Quuen Bee Present

    #Loop through NoQueenBee folder
for i in NoQB_files:
    #preprocess the audio file
    p = (os).path.join('Testing_data/NoQueenBee', i)

    audio, sample_rate = librosa.load(p, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        #Reshape MFCC feature to 2-D array since its one file
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        #Getting the predicted label for each input sample.
    x_predict=model.predict(mfccs_scaled_features)
    predicted_label=np.argmax(x_predict,axis=1)

    if predicted_label == 0:
        print('N0 Queen Bee')
    else:
        print('Queen Bee Present \n')

    #Loop through QueenBee_present folder
for i in QBP_files:
    #preprocess the audio file
    p = (os).path.join('Testing_data/QueenBee_present', i)

    audio, sample_rate = librosa.load(p, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        #Reshape MFCC feature to 2-D array since its one file
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        #Getting the predicted label for each input sample.
    x_predict=model.predict(mfccs_scaled_features)
    predicted_label=np.argmax(x_predict,axis=1)

    if predicted_label == 0:
        print('\n N0 Queen Bee')
    else:
        print('Queen Bee Present')


