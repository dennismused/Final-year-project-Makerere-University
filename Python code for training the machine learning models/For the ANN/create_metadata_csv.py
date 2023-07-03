    #CREATING A CSV FILE FOR THE AUDIO DATASET

import pandas as pd
import os
import librosa

    #Directories for audio files for a hive with no queen bee and a hive when a queen bee is present respectively
NoQB_files = 'Prepared beehive dataset/No queen bee'
QBP_files = 'Prepared beehive dataset/Queen bee present'

    #A function that gets files in the directory and returns them in an array
def getFilesInDir(directory):
    files = []
    files = os.listdir(directory)
    return files

    #Arrays of audio files for a hive with no queen bee and a hive when a queen bee is present respectively
NoQB_file_names = getFilesInDir(NoQB_files)
QBP_file_names = getFilesInDir(QBP_files)

    #Array of all audio files in both folders
file_names = NoQB_file_names + QBP_file_names


NoQB_number = len(NoQB_file_names) #Number of files under the No queen bee folder
QBP_number = len(QBP_file_names)   #Number of files under the Queen bee present folder
total_number = NoQB_number + QBP_number

    #Creating rows to fall under various column titles
sample_rate = ['22050']*total_number       #We multiply to apply it to other rows
label_NoQueen = ['No Queen']*NoQB_number
label_Queen = ['Queen Present']*QBP_number
label = label_NoQueen + label_Queen
length = [2]*total_number
NoQueen_folder = ['No queen bee']*NoQB_number
Queen_folder = ['Queen bee present']*QBP_number
folder = NoQueen_folder + Queen_folder

    #Creating a pandas dataframe from the above data and saving it as a csv file
df = pd.DataFrame({'filename':file_names, 'sr':sample_rate, 'label':label, 'length':length, 'folder':folder})
df.to_csv('E:/Users/DENNIS/PycharmProjects/pythonProject/Final-Year-Project/Prepared beehive dataset/metadata.csv')


