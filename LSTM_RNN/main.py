#!/usr/bin/python
import sys
from process_image import ProcessedImage, StringsToBatches, StringsToImage
from lstm_net import LSTMNet, RunNet

def RunAll():
	"""Run python main.py <name of image>"""
	#process image
	input_image = sys.argv[1]
	new_img = ProcessedImage(input_image)
	vocab_sz, pixelstr_ints, reversed_dict = StringsToBatches.words_into_ints(new_img.string_pixels)
	#set up training and generating nets
	training_net = LSTMNet("train", vocab_sz, .5, 50, 256, 20, 15, 5)
	generating_net = LSTMNet("generate", vocab_sz, 1.0, 50, 256, 1, 1)
	#get train and test data
	train_data, test_data = StringsToBatches.split_train_test(pixelstr_ints, .80)
	train_batches = StringsToBatches.get_batches(train_data, training_net.batch_size, training_net.num_steps)
	test_batches =  StringsToBatches.get_batches(test_data, training_net.batch_size, training_net.num_steps)
	#train, test, and generate
	run_nets = RunNet(training_net, generating_net, train_batches, test_batches, 102400)
	run_nets.train_net()
	run_nets.test_net()
	run_nets.generate_image()
	#create new image
	result_pixelstr = StringsToBatches.ints_to_pixelstr(reversed_dict, run_nets.final_pixels)
	pixels = [StringsToImage.string_pixel_to_pixel(w) for w in result_pixelstr]
	StringsToImage.string_pixel_to_image(pixels, new_img.width)

if __name__ == "__main__":
	RunAll()
