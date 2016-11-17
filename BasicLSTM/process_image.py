#!/usr/bin/python
import PIL.Image as Image
import numpy as np

class ProcessedImage():

	def __init__(self, filepath):
		self.image_location = "images"+ "/" + filepath
		self.width = None
		self.height = None
		self.string_pixels = None
		self.covert_image()

	def covert_image(self):
		img = Image.open(self.image_location)
		#limit colors to 256
		img_limited = img.convert('P', palette=Image.ADAPTIVE, colors=256)
		#convert to pixel tuples
		rgb_img = img_limited.convert("RGB")
		img_pixels  = list(rgb_img.getdata())
		width, height = rgb_img.size
		self.width = width
		self.height = height
		print "Uploaded image size is: {0} width X {1} height".format(width, height)
		flatten_pixels = [img_pixels[i * width:(i + 1) * width] for i in xrange(height)]
		string_pixels = ["r"+str(i[0])+"g"+str(i[1])+"b"+str(i[2]) for j in flatten_pixels for i in j]
		self.string_pixels = string_pixels
		print "Image to string processing complete"

class StringsToImage():

	@staticmethod
	def string_pixel_to_pixel(word):
		new_list = word.replace("r", "").replace("b", "|").replace("g", "|")
		new_list = new_list.split("|")
		final_pixel = (int(new_list[0]), int(new_list[1]), int(new_list[2]))
		return final_pixel

	@staticmethod
	def string_pixel_to_image(list_of_pixelstr, width):
		#print list_of_pixelstr, width
		num_full_rows = len(list_of_pixelstr) / width
		offset = len(list_of_pixelstr) - (num_full_rows * width)
		#print width, num_full_rows
		imgfinal = Image.new("RGB", ((width, num_full_rows)))
		imgfinal.putdata(list_of_pixelstr[offset:])
		imgfinal.save('testfinal.png')
		print "Test image saved to testfinal.png"

class StringsToBatches():

	@staticmethod
	def words_into_ints(list_of_pixelstr):
		word_to_int = dict()
		reversed_dict = dict()
		final_list = list()
		word_count = 0
		for word in list_of_pixelstr:
			if word in word_to_int:
				final_list.append(word_to_int[word])
			else:
				word_to_int[word] = word_count
				reversed_dict[word_count] = word
				word_count += 1
				final_list.append(word_to_int[word])
		return word_count+1, final_list, reversed_dict

	@staticmethod
	def split_train_test(data, percent_train):
		separator = int(round(len(data) * percent_train))
		train_data = data[:separator]
		test_data = data[separator:]
		return train_data, test_data

	@staticmethod
	def get_batches(data, batchsz, nsteps):
		start = 0
		step_sz = (batchsz * nsteps)
		end = step_sz
		batches = list()
		end_cap = (int(len(data)/end))
		i = 0
		while i < end_cap:
			i += 1
			if (len(data[start:end]) == step_sz) and (len(data[start+1:end+1]) == step_sz):
				batchx = np.reshape(data[start:end], (batchsz, nsteps))
				batchy = np.reshape(data[start+1:end+1], (batchsz, nsteps))
				batches.append((batchx, batchy))
				start = end
				end += step_sz
		return batches

	@staticmethod
	def ints_to_pixelstr(reversed_dict, data):
		return [reversed_dict[i] for i in data]
