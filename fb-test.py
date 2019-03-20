# -*- coding: UTF-8 -*-

import os
import time
from dotenv import load_dotenv
load_dotenv()
from fbchat import Client
from fbchat.models import *

client = Client(os.getenv("FBEMAIL"), os.getenv("FBPASS"))
myuser = client.fetchUserInfo(client.uid)[client.uid]

# Fetches a list of all users you're currently chatting with, as `User` objects
users = client.fetchAllUsers()
users.append(myuser)
print(users)

print("users' IDs: {}".format([user.uid for user in users]))
print("users' names: {}".format([user.name for user in users]))


# If we have a user id, we can use `fetchUserInfo` to fetch a `User` object
#user = client.fetchUserInfo(users[0].uid)[users[0].uid]
# We can also query both mutiple users together, which returns list of `User` objects
#users = client.fetchUserInfo(users[0].uid, users[0].uid)

#print("user's name: {}".format(user.name))
#print("users' names: {}".format([users[k].name for k in users]))


# `searchForUsers` searches for the user and gives us a list of the results,
# and then we just take the first one, aka. the most likely one:
#user = client.searchForUsers("samuel")[0]

#print("user ID: {}".format(user.uid))
#print("user's name: {}".format(user.name))
#print("user's photo: {}".format(user.photo))
#print("Is user client's friend: {}".format(user.is_friend))


# Fetches a list of the 20 top threads you're currently chatting with
threads = client.fetchThreadList()
# Fetches the next 10 threads
#threads += client.fetchThreadList(offset=20, limit=10)

#print("Threads: {}".format(threads))


# Gets the last 10 messages sent to the thread
messages = client.fetchThreadMessages(thread_id=threads[0].uid, limit=10)
# Since the message come in reversed order, reverse them
messages.reverse()

# Prints the content of all the messages
for message in messages:
	print([u.name for u in users if str(u.uid) == str(message.author)][0],": ", message.text)


# If we have a thread id, we can use `fetchThreadInfo` to fetch a `Thread` object
#thread = client.fetchThreadInfo(threads[0].uid)[threads[0].uid]
#print("thread's name: {}".format(thread.name))
#print("thread's type: {}".format(thread.type))


# `searchForThreads` searches works like `searchForUsers`, but gives us a list of threads instead
#thread = client.searchForThreads("lilian frances")[0]
#print("thread's name: {}".format(thread.name))
#print("thread's type: {}".format(thread.type))
