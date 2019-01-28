import numpy as np
import tensorflow as tf
try:
    import cPickle as pickle
except:
    import pickle
import os
import collections
import random
import sys
import py_midicsv
import tempfile
import csv

def read_midi_file(file_name,frequency):
    try:
        csv_string = py_midicsv.midi_to_csv(file_name)
    except:
        return []

    beats_per_second = 0

    heap_array = []

    for row in csv_string.getvalue().split('\n'):
        array_row = [x.strip() for x in row.split(',')]
        if array_row[2] == 'Header':
            beats_per_second = float(array_row[5])/frequency

        if array_row[2] == 'Note_on_c' and array_row[5] != 0:
            heap_element = [float(array_row[1])/beats_per_second, True, int(array_row[4])]
            heap_array.append(heap_element)

        if array_row[2] == 'Note_off_c' or (array_row[2] == 'Note_on_c' and array_row[5] == 0):
            heap_element = [float(array_row[1])/beats_per_second, False, int(array_row[4])]
            heap_array.append(heap_element)
    heap_array.sort()
    return heap_array

def to_numpy_array(array):
    max_time_value = int(array[len(array)-1][0])
    formated = np.zeros(shape=(max_time_value+1,128),dtype=np.bool)
    current_time = 0
    for x in range(0,max_time_value):
        if x != 0:
            formated[x] = formated[x-1]
        for y in range(current_time,len(array)):
            if array[y][0]//1 == x:
                formated[x][array[y][2]] = array[y][1]
                current_time = max(x,current_time)
            elif array[y][0] > x:
                continue
    return formated

def reverse_transformation(array,frequency,beats_per_sec):
    csv_file = open("gen.csv",'w')
    csv.register_dialect('registered', delimiter=',', quoting=csv.QUOTE_NONE, lineterminator = '\n')
    writer = csv.writer(csv_file,dialect='registered')
    freq = int(beats_per_sec/frequency)
    writer.writerow([0, 0, 'Header', 1, 3, beats_per_sec])
    writer.writerow([1, 0, 'Start_track'])
    for x in range(len(array)):
        if x == 0:
            for y in range(len(array[x])):
                if array[x][y]:
                    writer.writerow([1, x*freq, 'Note_on_c', 0, y , 65])
            continue
        for y in range(len(array[x])):
            if array[x][y] ^ array[x-1][y]:
                if array[x][y] == True:
                    writer.writerow([1, x*freq, 'Note_on_c', 0,y , 65])
                if array[x][y] == False:
                    writer.writerow([1,x*freq,'Note_off_c',0,y ,0])
    writer.writerow([1,(len(array)+1)*freq,'End_track'])
    writer.writerow([0,0,'End_of_file'])
    csv_file.close()
    csv_file = open("gen.csv",'r')
    midi_object = py_midicsv.csv_to_midi(csv_file)
    filename = "gen.mid"
    output_file = open(filename,'wb')
    midi_writer = py_midicsv.FileWriter(output_file)
    midi_writer.write(midi_object)
    
def array_to_int(array,start_note,features):
    if features <=16:
        int_array = np.zeros(shape=(len(array)),dtype = np.uint16)
    elif features <=32:
        int_array = np.zeros(shape=(len(array)),dtype = np.uint32)
    elif features <=64:
        int_array = np.zeros(shape=(len(array)),dtype = np.uint64)
    if features > 64:
        return []
    for x in range(len(array)):
        for y in range(start_note,start_note+features):
            if array[x][y]:
                int_array[x] += 2**(y-start_note)
    return int_array

def int_to_bool(array,start_note, features):
    bool_array = np.zeros(shape=(len(array),128),dtype=np.bool)
    for x in range(len(array)):
        i = start_note + features -1
        for bit in '{0:016b}'.format(array[x]):
            if bit == '1':
                bool_array[x][i] = True
            i -= 1
    return bool_array

def test():
    file_list = os.listdir(os.curdir)
    for file in file_list:
        if 'gen' not in file and '.mid' in file:
            array = read_midi_file(file,8)
            if array:
                formated = to_numpy_array(array)
                int_array = array_to_int(formated,58,16)
                reformated = int_to_bool(int_array, 58, 16)
                reverse_transformation(reformated,8,480)

def prepare_datafile(directory):
    file_list = os.listdir(r"C:\Users\Jakub\OneDrive\Eng\Source-midi\\" + directory)
    final_array = np.array([0],dtype = np.uint16)
    for file in file_list:
        if 'gen' not in file and '.mid' in file and '.csv' not in file:
            array = read_midi_file(r"C:\Users\Jakub\OneDrive\Eng\Source-midi\\" + directory + '\\' +file,8)
            if array:
                formated = to_numpy_array(array)
                int_array = array_to_int(formated,60,12)
                final_array = np.concatenate((final_array,int_array),axis=0)
    fp = open(directory+'.data','wb')
    pickle.dump(final_array,fp)
    fp.close()

def prepare_data(directory):
    file_list = os.listdir(r"C:\Users\Jakub\OneDrive\Eng\Source-midi\\" + directory)
    final_array = list()
    for file in file_list:
        if 'gen' not in file and '.mid' in file and '.csv' not in file:
            array = read_midi_file(r"C:\Users\Jakub\OneDrive\Eng\Source-midi\\" + directory + '\\' +file,8)
            if array:
                formated = to_numpy_array(array)
                final_array.append(formated)
    fp = open('test.data','wb')
    pickle.dump(final_array,fp)
    fp.close()


def translate():
    
    fp = open('gen (1).data','rb')
    fp.seek(0)
    array = pickle.load(fp)
    reverse_transformation(int_to_bool(array,60,12),8,480)

if __name__ == '__main__':
    midis = list()
    midis.append(to_numpy_array(read_midi_file('ashover29.mid',8)))
    midis.append(to_numpy_array(read_midi_file('ashover30.mid',8)))
    for song in midis:
        int_array = array_to_int(song,58,16)
        bool_array = int_to_bool(int_array,58,16)
        for first_value, second_value in zip(song, bool_array):
            print(np.array_equal(first_value[58:58+16],second_value[58:58+16]))


