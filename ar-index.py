import os
import time
from dotenv import load_dotenv
load_dotenv()
from random import random
import re
import sys
import numpy as np
import sched, time
import urllib.request
import getpass

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils

from fbchat import log, Client
from fbchat.models import *

def getRandomMeme():
	try:
		contents = str(urllib.request.urlopen("https://knowyourmeme.com/memes/random").read())
		ti = contents.find('https://i.kym-cdn.com/entries/icons') # find the start of an icon on the page
		tti = contents[ti:].find('.jpg') # find the end of that src
		return contents[ti:ti+tti+4] # return src
	except:
		return getRandomMeme()

class EchoBot(Client):
	def reloadCharToInt(self):
		self.char_to_int = dict((c, i) for i, c in enumerate(self.chars))
		self.int_to_char = dict((i, c) for i, c in enumerate(self.chars))
		self.nvocab = len(self.chars)

	def getInitials(self, name):
		name = self.simplifyMessage(name)
		arName = name.split(' ')
		return arName[0][0] + arName[1][0]

	def loadBrain(self, brainFile):
		self.myuser = self.fetchUserInfo(self.uid)[self.uid]

		# Fetches a list of all users you're currently chatting with, as `User` objects
		self.users = self.fetchAllUsers()
		self.users.append(self.myuser)

		#self.chars = ['\n', '\r', ' ', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '@', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']
		self.chars = ['\n', ' ', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '>', '[', '\\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']
		self.reloadCharToInt()

		self.model = Sequential()
		self.model.add(LSTM(256, input_shape=(200, 1), return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(256))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(self.nvocab, activation='softmax'))
		
		filename = os.path.join(os.getcwd(), 'brains', brainFile)
		self.model.load_weights(filename)

		self.model.compile(loss='categorical_crossentropy', optimizer='adam')

	def simplifyMessage(self, txt):
		ftext = ''.join(''.join(''.join(re.split(r"[^A-z0-9\s,.:;<>\-\+_]", txt.lower())).split('\x85')).split('~'))
		tftext = ""
		for i in ftext:
			if i in self.chars:
				tftext += i
		return tftext

	def useBrain(self, inp):
		print("starting to compute a new message...")

		inp = self.simplifyMessage(inp)
		inp = inp[:200]
		while len(inp) < 200:
			inp = " " + inp
		pattern = [self.char_to_int[char] for char in inp]

		tout = ""
		tildes = 2

		for i in range(200):
			x = np.reshape(pattern, (1, len(pattern), 1))
			x = x / float(self.nvocab)
			prediction = self.model.predict(x, verbose=0)
			
			index = np.argmax(prediction)
			result = self.int_to_char[index]
			seq_in = [self.int_to_char[value] for value in pattern]
			tout += result
			sys.stdout.write("\n"+result)

			if result == '~':
				tildes -= 1
				if tildes <= 0:
					return tout

			pattern.append(index)
			pattern = pattern[1:len(pattern)]

		print("computed new message", tout)

		return tout

	def parseAiMessage(self, message):
		tok = message.split('~')
		for msg in tok:
			if msg[:2] == 'sc': # if its my message
				tmsg = msg[2:]
				return tmsg
		return False


	def simplifyMessages(self, messages):
		tout = ""
		messages.reverse()
		for message in messages:
			smsg = self.simplifyMessage(message.text)+"\n"
			tout += "~"+str([self.getInitials(u.name) for u in self.users if str(u.uid) == str(message.author)][0])+smsg

		return tout
	
	def onMessage(self, author_id, message_object, thread_id, thread_type, **kwargs):
		self.markAsDelivered(thread_id, message_object.uid)
		self.markAsRead(thread_id)

		log.info("{} from {} in {}".format(message_object, thread_id, thread_type.name))

		# If you're not the author, echo
		if author_id != self.uid:
			if message_object.text == None:
				message_object.text = ''
			if len(message_object.text) > 6:
				tmsgs = self.fetchThreadMessages(thread_id=thread_id, limit=15)
				simpleMsgs = self.simplifyMessages(tmsgs)
				print(simpleMsgs)
				tmsg = self.parseAiMessage(self.useBrain(simpleMsgs))
				if 'you sent a photo.' in tmsg or 'you sent a sticker.' in tmsg:
					turl = getRandomMeme()
					print(turl)
					self.sendRemoteImage(turl, thread_id=thread_id, thread_type=thread_type)
				else:
					message_object.text = tmsg
					self.send(message_object, thread_id=thread_id, thread_type=thread_type)
			else:
				rexp = re.compile(r'U\+([0-9A-Za-z]+)[^#]*# [^)]*\) *(.*)')
				m = rexp.match(message_object.text)
				if m or message_object.emoji_size != None or message_object.sticker != None:
					print('received an emoji')
					self.send(Message(emoji_size=EmojiSize.LARGE), thread_id=thread_id, thread_type=thread_type)

tpass = getpass.getpass('Please input the password for email <'+str(os.getenv('FBEMAIL'))+'>\n ~ ')
client = EchoBot(os.getenv('FBEMAIL'), tpass)
client.loadBrain(os.path.join('sammain.hdf5'))
client.listen()
#for i in range(0, 10):
#	print(getRandomMeme())
