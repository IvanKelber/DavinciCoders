#!/usr/bin/python
import tensorflow as tf
import numpy as np

class BasicLSTMNet():

	#weights and biases are class scope, not instance scope.
	w1 = None
	b1 = None

	def __init__(self, net_name, vocab_size, k_prob, e_size, hsize, num_steps, batch_size, epochs=None):
		self.k_prob = k_prob
		self.e_size = e_size
		self.hsize = hsize
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.epochs = epochs
		self.vocab_size = vocab_size
		self.net_name = net_name
		self.setup_net()

	def setup_net(self):
		with tf.variable_scope(self.net_name):
			self.inputs = tf.placeholder(tf.int32, shape=[None, self.num_steps])
			self.labels = tf.placeholder(tf.int32, shape=[None, self.num_steps])
			self.keep_prob = tf.placeholder(tf.float32, shape=None)
			embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.e_size], -.1, .1), name="embeddings")
			embedding_layer = tf.nn.embedding_lookup(embeddings, self.inputs)
			embedding_layer = tf.nn.dropout(embedding_layer, self.k_prob)
			self.b_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hsize, state_is_tuple=True)
			self.init_state = self.b_cell.zero_state(self.batch_size, tf.float32)
			#this is not the best way to do this
			if self.__class__.w1 == None and self.__class__.b1 == None:
				self.__class__.w1 = tf.Variable(tf.random_uniform([self.hsize, self.vocab_size], -.1, .1))
				self.__class__.b1 = tf.Variable(tf.random_uniform([self.vocab_size], -.1, .1))
			outputs, self.f_state = tf.nn.dynamic_rnn(self.b_cell, embedding_layer, initial_state=self.init_state)
			outputs = tf.reshape(outputs, [(self.batch_size * self.num_steps), self.hsize])
			logits = tf.matmul(outputs, self.w1) + self.b1
			self.prediction = logits
			useless_weights = tf.ones([self.batch_size * self.num_steps])
			self.error = tf.reduce_sum(tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.labels, [self.batch_size*self.num_steps])], [useless_weights]))
			avg_loss = self.error/self.batch_size
			self.train_step = tf.train.AdamOptimizer(.0001).minimize(self.error)
			self.sess = tf.Session()
			self.sess.run(tf.initialize_all_variables())
			print "Basic LSTM net initialized"

class RunNet():

	def __init__(self, training_net, generating_net, training_batches, test_batches, pixels_to_gen):

		self.training_net = training_net
		self.generating_net = generating_net
		self.training_batches = training_batches
		self.test_batches = test_batches
		self.final_pixels = list()
		self.starter_pixel = test_batches[0][0][0][0]
		self.pixels_to_gen = pixels_to_gen
	
	def train_net(self):

		for e in range(self.training_net.epochs):
			print "Current epoch:", e		
			cur_state = (self.training_net.init_state.c.eval(session=self.training_net.sess), self.training_net.init_state.h.eval(session=self.training_net.sess))
			for ix, b in enumerate(self.training_batches):
				feed_dict = {self.training_net.inputs:b[0], self.training_net.labels:b[1], self.training_net.keep_prob:[self.training_net.k_prob], self.training_net.init_state[0]:cur_state[0], self.training_net.init_state[1]:cur_state[1]}
				_, cur_state, err = self.training_net.sess.run([self.training_net.train_step, self.training_net.f_state, self.training_net.error], feed_dict=feed_dict)
				if ix%100 == 0:
					print "Current perplexity at batch {0}:".format(ix), np.exp(err / (self.training_net.batch_size * self.training_net.num_steps))
		print "Training complete"

	def __getWord(self, l):
		pixels = list()
		for logits in l:
				get_max = max(logits)
				pixels.append([i for i, j in enumerate(logits) if j == get_max][0])
		return pixels

	def test_net(self):

		k_prob = 1.0
		init_state = self.training_net.b_cell.zero_state(self.training_net.batch_size, tf.float32)
		cur_state = (init_state.c.eval(session=self.training_net.sess), init_state.h.eval(session=self.training_net.sess))
		final_error = 0

		for b in self.test_batches:
			feed_dict = feed_dict = {self.training_net.inputs:b[0], self.training_net.labels:b[1], self.training_net.keep_prob:[k_prob], init_state[0]:cur_state[0], init_state[1]:cur_state[1]}
			cur_state, err = self.training_net.sess.run([self.training_net.f_state, self.training_net.error], feed_dict=feed_dict)
			final_error += (err / (self.training_net.batch_size * self.training_net.num_steps))
		print 'final error:', np.exp((final_error / (len(self.test_batches))))
		print 'Testing complete'

	def generate_image(self):

		cur_state_gen = (self.generating_net.init_state.c.eval(session=self.generating_net.sess), self.generating_net.init_state.h.eval(session=self.generating_net.sess))
		num_generated = 0
		while num_generated < self.pixels_to_gen:
			
			previous_pixel = np.array([[self.starter_pixel]], dtype=np.int32)
			feed_dict = {self.generating_net.inputs:previous_pixel, self.generating_net.keep_prob:[self.generating_net.k_prob], self.generating_net.init_state[0]:cur_state_gen[0], self.generating_net.init_state[1]:cur_state_gen[1]}
			pred_gen, cur_state_gen = self.generating_net.sess.run([self.generating_net.prediction, self.generating_net.f_state], feed_dict=feed_dict)
			cur_p = self.__getWord(pred_gen)[0]
			self.starter_pixel = cur_p
			self.final_pixels.append(cur_p)
			num_generated += 1
			if num_generated % 1000 == 0:
				print "Pixels generated:", num_generated
