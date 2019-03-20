import os
import re
import numpy
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # comment this out to enable gpu

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils

rawtext = ''
maxFiles = 90
fileNames = [os.path.join('songs', f) for f in os.listdir('songs')]
for f in fileNames:
	if maxFiles > 0:
		ftext = ''.join(''.join(re.split(r"[^A-z0-9\s,.:;<>\-\+_]", open(f, encoding="ISO-8859-1").read().lower())).split('\x85'))
		if len(ftext) > 10:
			rawtext += ftext
			maxFiles -= 1
			#if ftext.find('les ') != -1:
				#print(f, ftext.find('les '))
rts = list(set(rawtext))

nchars = len(rawtext)
nvocab = len(rts)

chars = sorted(rts)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

seq_length = 100
dataX = [] # note for future readers, X/Y is a common tf/keras abstraction. X is input, Y is output (desired)
dataY = []
for i in range(0, nchars - seq_length, 1): #split input data into sequences of seq_length
	seq_in = rawtext[i:i+seq_length]
	seq_out = rawtext[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

npatterns = len(dataX)

print("loaded " + str(len(rawtext)) + " chars of rap music, with vocab of size " + str(nvocab) + " and " + str(seq_length) + " seqlength")
print("vocab: " + str(rts))

X = numpy.reshape(dataX, (npatterns, seq_length, 1))
X = X / float(nvocab) # normalize
y = np_utils.to_categorical(dataY) # one hot encode the output variable

# BIG BOY MODEL TIME!!!!! #

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filename = os.path.join('brains', "weights-improvement-35-1.6267.hdf5")
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint(s)
filepath=os.path.join("brains", "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=45, batch_size=50, callbacks=callbacks_list)


##
print("\n---\nDONE MOTHA FUCKIN TRAINING YALL\n---")
##

# get a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for i in range(200):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(nvocab)
	prediction = model.predict(x, verbose=0)
	
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)

	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print("\n---\nDone.\n---")
