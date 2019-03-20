import os
import re
import numpy as np
import sys
import json
from random import shuffle as rshuffle
from random import choice as rchoice
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # comment this out to enable gpu

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils

TRAINING_DATASET = "fb" # fb or reddit

MAX_CHATS = 25
MAX_CHAT_LENGTH = 210500 # only for use with training_dataset "fb"
MAX_CHAT_MESSAGES = 415

def simplifyMessage(txt):
	ftext = ''.join(re.split(r'[\r\t\`\x85]', ''.join(re.split(r'[^A-z0-9\s,.:;<>\-\+_]', txt.lower()))))
	return ftext

def loadSimpleJson(f):
	data = False

	with open(f) as json_data:
		data = json.load(json_data)
	if not data or len(data) <= 0:
		print("FUCKING DATA NOT FOUND!", f)
		exit()

	return data

def getInitials(name):
	name = simplifyMessage(name)
	arName = name.split(' ')
	return arName[0][0] + arName[1][0]

def isMe(name):
	if 'sam' in name and 'clarke' in name:
		return 'm'
	else:
		return 'y'

def loadSimpleMessageSet(dat):
	tmsg = ""
	dat['messages'].reverse()
	ctr = 0
	for i in dat['messages']:
		if ctr < MAX_CHAT_MESSAGES:
			if i['type'] == 'Generic':
				if not 'content' in i:
					i['content'] = ''
				tmsg += "~" + getInitials(simplifyMessage(i['sender_name']))+ simplifyMessage(i["content"]) + "\n"
				ctr += 1
	return tmsg + "@"

def readFBDataSet():
	fileNames = [os.path.join('datasets', 'messages', f, 'message.json') for f in os.listdir(os.path.join('datasets', 'messages'))]
	messages = ""

	while len(messages) < MAX_CHAT_LENGTH:
		messages += loadSimpleMessageSet(loadSimpleJson(rchoice(fileNames)))

	print(len(messages), ' chars loaded for trainings')

	return messages

def readRedditDataSet():
	corpus = loadSimpleJson(os.path.join(os.getcwd(), 'datasets', 'redditconvos.json'))

	print(str(len(corpus)) + " results loaded.")

	corpusByRoot = {}
	flatCorpus = ""

	for i in corpus:
		if len(corpusByRoot) < MAX_CHATS:
			if not i['user-info']['user-deleted'] and i['text'] != '[deleted]':
				if not i['root'] in corpusByRoot:
					corpusByRoot[i['root']] = []
				corpusByRoot[i['root']].append(i)

	for root in corpusByRoot:
		corpusByRoot[root].sort(key=lambda x: x['timestamp'])
		for msg in corpusByRoot[root]:
			flatCorpus += simplifyMessage(msg['user']) + '~' + simplifyMessage(msg['text']) + "\n"
		flatCorpus += "@\n"
	flatCorpus = flatCorpus[:-2] # chop off the last "@\n"

	return flatCorpus

print("-= STARTED READING DATASETS =-")

rawtext = ""
if TRAINING_DATASET == "reddit":
	rawtext = readRedditDataSet()
elif TRAINING_DATASET == "fb":
	rawtext = readFBDataSet()

if rawtext == "":
	print("invalid training dataset selected, please use either 'reddit' or 'fb'")
	exit()

print("-= FINISHED READING DATASETS =-")

rts = list(set(rawtext))
rts.remove('@')

nchars = len(rawtext)
nvocab = len(rts)

chars = ['\n', ' ', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '>', '[', '\\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']#sorted(rts)
print(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

#exit()

seq_length = 200
dataX = [] # note for future readers, X/Y is a common tf/keras abstraction. X is input, Y is output (desired)
dataY = []
for i in range(0, nchars - seq_length, 1): #split input data into sequences of seq_length
	seq_in = rawtext[i:i+seq_length]
	seq_out = rawtext[i + seq_length]
	if '@' in seq_in or '@' in seq_out:
		#print('oopsie there was an @ in ', seq_in, seq_out)
		continue
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

npatterns = len(dataX)

print("loaded " + str(len(rawtext)) + " chars of chats, with vocab of size " + str(nvocab) + " and " + str(seq_length) + " seqlength")
print("vocab: " + str(rts))

X = np.reshape(dataX, (npatterns, seq_length, 1))
X = X / float(nvocab) # normalize
y = np_utils.to_categorical(dataY) # one hot encode the output variable

# BIG BOY MODEL TIME!!!!! #

print('XShape 1', X.shape[1], 'XShape 2', X.shape[2], 'yshape 1', y.shape[1])
#exit()

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filename = os.path.join('brains', 'trainmain.hdf5')
model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint(s)
filepath=os.path.join("brains", "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=10, batch_size=86, callbacks=callbacks_list)


##
print("\n---\nDONE MOTHA FUCKIN TRAINING YALL\n---")
##


# get a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for i in range(200):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(nvocab)
	prediction = model.predict(x, verbose=0)
	
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)

	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print("\n---\nDone.\n---")
