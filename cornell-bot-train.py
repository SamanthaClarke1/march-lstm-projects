import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import time
import re
import random
import tensorflow as tf

#okay this is pretty much all this one dudes nn, and he writes the entire thing WITHOUT keras, which fucking scares me tbh
#https://tutorials.botsfloor.com/how-to-build-your-first-chatbot-c84495d4622d
#AAAAAAAAAAAA
#might not even use this just cos raw tf is hard
#idk, we'll see

def model_inputs():
	#creates placeholders for our model's inputs (THIS ISNT KERAS, AND I KNOW RAW TF IS SCARY)
	input_data = tf.placeholder(tf.int32, [None, None], name='input')
	targets = tf.placeholder(tf.int32, [None, None], name='targets')

	lr = tf.placeholder(tf.float32, name='learning_rate')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):
	#Remove the last word id from each batch and concat the <GO> to the begining of each batch
	ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
	dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
	#this formatting is neccessary for creating the embeddings for our decoding layer
	return dec_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
	#encodes our input data
	lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

	drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)

	enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

	_, enc_state = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=enc_cell,
				cell_bw=enc_cell,
				sequence_length=sequence_length,
				inputs=rnn_inputs, 
				dtype=tf.float32)

	#LSTM > GRU for seq2seq, like this NN. Making the encoder bidirectional is better than feed forward.
	#We only return encoders state because its the input for our decoding layer 
	#simply put, the weights of the encoding cells are what we care about

	return enc_state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
							sequence_length, decoding_scope,
							output_fn, keep_prob, batch_size):

	#using attention in our decoding layers lowers the loss by ~20%, decent trade-off
	#model performs best when attention states are zeros
	#two attention options are bahdanau and luong. bahdanau is faster, and seems to work better.
	attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

	att_keys, att_vals, att_score_fn, att_construct_fn = \
		tf.contrib.seq2seq.prepare_attention(
				attention_states,
				attention_option="bahdanau",
				num_units=dec_cell.output_size)

	train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(
				encoder_state[0],
				att_keys,
				att_vals,
				att_score_fn,
				att_construct_fn,
				name="attn_dec_train")

	train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
				dec_cell, 
				train_decoder_fn, 
				dec_embed_input, 
				sequence_length, 
				scope=decoding_scope)

	train_pred_drop = tf.nn.dropout(train_pred, keep_prob)

	return output_fn(train_pred_drop)

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
							start_of_sequence_id, end_of_sequence_id,
							maximum_length, vocab_size, decoding_scope,
							output_fn, keep_prob, batch_size):
	#honestly just wish i could be using keras right about now :/

	#this fn is a lot like decode_layer_train, main difference is more params in decoder_fn_inference and decoder_fn_train
	#they're neccessary to help the model create accurate responses for input sentences
	attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

	att_keys, att_vals, att_score_fn, att_construct_fn = \
		tf.contrib.seq2seq.prepare_attention(
			attention_states,
			attention_option="bahdanau",
			num_units=dec_cell.output_size)

	infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(
			output_fn, 
			encoder_state[0], 
			att_keys, 
			att_vals, 
			att_score_fn, 
			att_construct_fn, 
			dec_embeddings,
			start_of_sequence_id, 
			end_of_sequence_id, 
			maximum_length, 
			vocab_size, 
			name = "attn_dec_inf")

	infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
		dec_cell, 
		infer_decoder_fn, 
		scope=decoding_scope)

	return infer_logits

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, 
					vocab_size, sequence_length, rnn_size,
					num_layers, vocab_to_int, keep_prob, batch_size):
	#here we use the prev 2 fn's, a decoding cell, and a fully connected layer to create our training and inference logits
	#we use tf.variable_scope to reuse the variables from training for making predictions
	#its encouraged to init weights and biases, by init'ing them with a truncated normal distribution and a small standard
	#deviation, this can really help to improve the performance of your model.
	with tf.variable_scope("decoding") as decoding_scope:
		lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
		drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
		dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

		weights = tf.truncated_normal_initializer(stddev=0.1)
		biases = tf.zeros_initializer()
		#AH! LAMBDAS IN PYTHON! FUCK, SCARY, STOP. STOP! SCARY!
		output_fn = lambda x: tf.contrib.layers.fully_connected(
						x, 
						vocab_size, 
						None, 
						scope=decoding_scope,
						weights_initializer = weights,
						biases_initializer = biases)

		train_logits = decoding_layer_train(encoder_state, 
											dec_cell, 
											dec_embed_input, 
											sequence_length, 
											decoding_scope, 
											output_fn, 
											keep_prob, 
											batch_size)
		decoding_scope.reuse_variables()

		infer_logits = decoding_layer_infer(encoder_state, 
											dec_cell, 
											dec_embeddings, 
											vocab_to_int['<GO>'],
											vocab_to_int['<EOS>'], 
											sequence_length - 1, 
											vocab_size,
											decoding_scope, 
											output_fn, 
											keep_prob, 
											batch_size)

	return train_logits, infer_logits

def seq2seq_model(input_data, target_data, keep_prob, batch_size, 
					sequence_length, answers_vocab_size, 
					questions_vocab_size, enc_embedding_size, 
					dec_embedding_size, rnn_size, num_layers, 
					questions_vocab_to_int):
	#this is where we tie the model together (duh) and generate outputs for our model.
	#similar to init'ing weights and biases, init your embeddings too.
	#rather than a truncated normal distribution, a random uniform distribution is more appropriate.
	#read more here https://www.tensorflow.org/tutorials/word2vec

	#we use tf.contrib.layers.embed_sequence() to simplify the code a little. (about time) if you ask me.

	enc_embed_input = tf.contrib.layers.embed_sequence(
		input_data, 
		answers_vocab_size+1, 
		enc_embedding_size,
		initializer = tf.random_uniform_initializer(-1,1))

	enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

	dec_input = process_encoding_input(target_data,      
										questions_vocab_to_int, 
										batch_size)
	dec_embeddings = tf.Variable(  
		tf.random_uniform([questions_vocab_size+1,  
							dec_embedding_size], 
							-1, 1))

	dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, 
												dec_input)

	train_logits, infer_logits = decoding_layer(
		dec_embed_input, 
		dec_embeddings, 
		enc_state, 
		questions_vocab_size, 
		sequence_length, 
		rnn_size, 
		num_layers, 
		questions_vocab_to_int, 
		keep_prob, 
		batch_size)

	return train_logits, infer_logits

epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

min_line_length = 2
max_line_length = 20
answers_vocab_to_int = {}
questions_vocab_to_int = {}

#sets up the structure of our graph
tf.reset_default_graph()
#uses an interactive session for flexibility, but you can use whatever really.
sess = tf.InteractiveSession()

input_data, targets, lr, keep_prob = model_inputs()
sequence_length = tf.placeholder_with_default(max_line_length, None, name='sequence_length')
input_shape = tf.shape(input_data)

train_logits, inference_logits = seq2seq_model(
	tf.reverse(input_data, [-1]), 
	targets, 
	keep_prob, 
	batch_size, 
	sequence_length, 
	len(answers_vocab_to_int), 
	len(questions_vocab_to_int), 
	encoding_embedding_size, 
	decoding_embedding_size, 
	rnn_size, 
	num_layers, 
	questions_vocab_to_int)

with tf.name_scope("optimization"):
	cost = tf.contrib.seq2seq.sequence_loss(train_logits, targets, tf.ones([input_shape[0], sequence_length]))

	optimizer = tf.train.AdamOptimizer(learning_rate)

	gradients = optimizer.compute_gradients(cost)
	capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
	train_op = optimizer.apply_gradients(capped_gradients)

#input_question = 'How are you?'

random = np.random.choice(len(short_questions))
input_question = short_questions[random]

input_question = question_to_seq(input_question, questions_vocab_to_int)

input_question = input_question + [questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))

batch_shell = np.zeros((batch_size, max_line_length))
batch_shell[0] = input_question

answer_logits = sess.run(inference_logits, {input_data: batch_shell, keep_prob: 1.0})[0]

pad_q = questions_vocab_to_int["<PAD>"]
pad_a = answers_vocab_to_int["<PAD>"]

print('Question')
print('  Word Ids: {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words: {}'.format([questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('Word Ids: {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('Response Words: {}'.format([answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))