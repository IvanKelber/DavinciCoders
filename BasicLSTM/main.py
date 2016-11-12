#!/usr/bin/python
import sys
from process_image import ProcessedImage, StringsToBatches, StringsToImage
from basic_LSTM_net import BasicLSTMNet

def RunAll():
	"""Run python mina.py <name of image>"""
	input_image = sys.argv[1]
	new_img = ProcessedImage(input_image)
	vocab_sz, pixelstr_ints, reversed_dict = StringsToBatches.words_into_ints(new_img.string_pixels)

	simple_net = BasicLSTMNet(vocab_sz, .5, 50, 256, 20, 35, 15)
	train_data, test_data = StringsToBatches.split_train_test(pixelstr_ints, .9)
	train_batches = StringsToBatches.get_batches(train_data, simple_net.batch_size, simple_net.num_steps)
	test_batches =  StringsToBatches.get_batches(test_data, simple_net.batch_size, simple_net.num_steps)
	
	simple_net.setup_net()
	simple_net.train_net(train_batches)
	simple_net.test_net(test_batches)

	result_pixelstr = StringsToBatches.ints_to_pixelstr(reversed_dict, simple_net.final_pixels)
	pixels = [StringsToImage.string_pixel_to_pixel(w) for w in result_pixelstr]
	StringsToImage.string_pixel_to_image(pixels, new_img.width)

if __name__ == "__main__":
	RunAll()