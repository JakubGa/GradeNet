import csv
import pickle
import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

file_list = os.listdir(os.curdir)
start_note = 58
features = 16

def translate(one_file):
    my_reader = csv.reader(open(one_file),delimiter = ',')
    note_on = list()
    note_of = list()
    for row in my_reader:
        if 'Header' in row[2]:
            note_length = int(row[5])/8
        elif row[2] == ' Note_on_c' and int(row[5]) != 0:
            note_on.append([int(int(row[1])/note_length),int(row[4])])
        elif row[2] == ' Note_off_c':
            note_of.append([int(int(row[1])/note_length),int(row[4])])
        elif row[2] == ' Note_on_c' and int(row[5]) == 0:
            note_of.append([int(int(row[1])/note_length),int(row[4])])
    return note_on, note_of

def get_array(list_on,list_of):
    iterable_length = max(len(list_on),len(list_of))
    list_on.sort()
    list_of.sort()
    try:
        music_length = list_of[iterable_length-1][0]
    except:
        music_length = list_on[iterable_length-1][0]
    music = np.zeros(shape=(music_length+1),dtype=np.uint8)
    for row in list_on:
        if row[1] > start_note-1 and row[1] < start_note+features:
            music[row[0]] += pow(2,row[1]-start_note)
    for row in list_of:
        if row[1] > start_note-1 and row[1] < start_note+features:
            music[row[0]] -= pow(2,row[1]-start_note)
    for value in range(1,music_length-1):
        music[value] = music[value]^music[value-1]
    return music

fp = open('httpswww.ics.uci.edu~danmidirock.data','rb')
final_array = pickle.load(fp)
fp.close()

seq_length = 64
examples_per_epoch = len(final_array)//seq_length

final_array = tf.data.Dataset.from_tensor_slices(final_array)

sequences = final_array.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
  
# Batch size 
BATCH_SIZE = 2
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# Length of the vocabulary in chars
vocab_size = 2**16

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNLSTM
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.LSTM, recurrent_activation='sigmoid')


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = 2**16, 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

model.summary()

  
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)



model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)


# Directory where the checkpoints will be saved
checkpoint_dir = os.curdir + '\training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS=1


history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])