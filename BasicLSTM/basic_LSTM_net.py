#!/usr/bin/python
import tensorflow as tf
import numpy as np

class BasicLSTMNet():

	def __init__(self, vocab_size, k_prob, e_size, hsize, num_steps, batch_size, epochs):
		self.k_prob = k_prob
		self.e_size = e_size
		self.hsize = hsize
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.epochs = epochs
		self.vocab_size = vocab_size

	def setup_net(self):
		self.inputs = tf.placeholder(tf.int32, shape=[None, self.num_steps])
		self.labels = tf.placeholder(tf.int32, shape=[None, self.num_steps])
		self.keep_prob = tf.placeholder(tf.float32, shape=None)
		state1 = tf.placeholder(tf.float32, shape=[self.hsize])
		state2 = tf.placeholder(tf.float32, shape=[self.hsize])
		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.e_size], -.1, .1), name="embeddings")
		embedding_layer = tf.nn.embedding_lookup(embeddings, self.inputs)
		embedding_layer = tf.nn.dropout(embedding_layer, self.k_prob)
		self.b_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hsize, state_is_tuple=True)
		self.init_state = self.b_cell.zero_state(self.batch_size, tf.float32)
		w1 = tf.Variable(tf.random_uniform([self.hsize, self.vocab_size], -.1, .1))
		b1 = tf.Variable(tf.random_uniform([self.vocab_size], -.1, .1))
		outputs, self.f_state = tf.nn.dynamic_rnn(self.b_cell, embedding_layer, initial_state=self.init_state)
		outputs = tf.reshape(outputs, [(self.batch_size * self.num_steps), self.hsize])
		logits = tf.matmul(outputs, w1) + b1
		self.prediction = logits
		useless_weights = tf.ones([self.batch_size * self.num_steps])
		self.error = tf.reduce_sum(tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.labels, [self.batch_size*self.num_steps])], [useless_weights]))
		avg_loss = self.error/self.batch_size
		self.train_step = tf.train.AdamOptimizer(.0001).minimize(self.error)
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		print "Basic LSTM net initialized"

	def train_net(self, batches):

		for e in range(self.epochs):
			print "Current epoch:", e
			init_state = self.b_cell.zero_state(self.batch_size, tf.float32)
			cur_state = (init_state.c.eval(session=self.sess), init_state.h.eval(session=self.sess))
			for ix, b in enumerate(batches):
				feed_dict = {self.inputs:b[0], self.labels:b[1], self.keep_prob:[self.k_prob], init_state[0]:cur_state[0], init_state[1]:cur_state[1]}
				_, cur_state, err = self.sess.run([self.train_step, self.f_state, self.error], feed_dict=feed_dict)
				if ix%100 == 0:
					print "Current perplexity at batch {0}:".format(ix), np.exp(err / (self.batch_size * self.num_steps))

	def test_net(self, batches):

		k_prob = 1.0
		init_state = self.b_cell.zero_state(self.batch_size, tf.float32)
		cur_state = (init_state.c.eval(session=self.sess), init_state.h.eval(session=self.sess))
		final_error = 0

		def getWord(l):
			pixels = list()
			for logits in l:
					get_max = max(logits)
					pixels.append([i for i, j in enumerate(logits) if j == get_max][0])
			return pixels

		self.final_pixels = list()

		for b in batches:
			feed_dict = feed_dict = {self.inputs:b[0], self.labels:b[1], self.keep_prob:[k_prob], init_state[0]:cur_state[0], init_state[1]:cur_state[1]}
			pred, cur_state, err = self.sess.run([self.prediction, self.f_state, self.error], feed_dict=feed_dict)
			self.final_pixels += getWord(pred)
			final_error += (err / (self.batch_size * self.num_steps))

		print 'final error:', np.exp((final_error / (len(batches))))
