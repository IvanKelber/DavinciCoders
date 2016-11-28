import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join
import cPickle
from PIL import Image

DIMENSION = 64
class Parser:
    dimension = DIMENSION

    def __init__(self, path, simple_images=False, cifar_flag=False):
        self.images = []
        self.final_images = []
        if cifar_flag:
            img_dict = self.unpickle(path)
            for ix, label in enumerate(img_dict['labels']):
                 if label == 0:
                    self.final_images.append(self.get_cifar_img(img_dict['data'][ix]))
            #Slice list so that it is a multiple of batch size, batch size must be square number
            self.final_images = self.final_images[:1000]
            self.final_images = np.reshape(self.final_images, (1000, -1))        
        elif simple_images:
            self.load_images_into_tensor_array(self.simple_get_images(path)) 
        else:
            for era in os.listdir(path):
                if(not isfile(era)):
                    self.images += self.get_images(path,era)
            self.load_images_into_tensor_array(self.images)
        

    def get_images(self,path,era):
        """
        Returns all of the images associated with that era in a list of tuples of form:
        (image_filename,path_to_era)
        """
        artists = [artist for artist in os.listdir(join(path,era)) if not isfile(join(path,era,artist))]
        images = []
        for artist in artists:
            images += [(join(path,era,artist,image),era) for image in os.listdir(join(path,era,artist))]
        return images

    def simple_get_images(self, path):
        """
        Used if images are all located in single folder, mainly for testing
        """
        images=[]
        for dir_item in os.listdir(path):
            if dir_item.endswith("jpg") or dir_item.endswith("jpeg"):
                images.append((join(path,dir_item), ""))
        return images

    def parse_jpg(jpg):
        return tf.image.decode_jpeg(jpg,channels=3,ratio=8)

    def unpickle(self, path):
        fo = open(path, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def scaled_rgb_tuple_to_grayscale(self, img):
        return round(int(round((.21 * img[0]) + (.72 * img[1]) + (.07 * img[2])))/255.0, 8)

    def get_cifar_img(self, raw_array):
        img = []
        index_r = 0
        index_g = 1024
        index_b = 2048
        while index_r < 1024:
            tup = (raw_array[index_r], raw_array[index_g], raw_array[index_b])
            gray = self.scaled_rgb_tuple_to_grayscale(tup)
            img.append(gray)
            index_r += 1
            index_g += 1
            index_b += 1
        img = np.reshape(img, (1024, -1))
        return img

    def pixel_to_image(self, array, width, height):
        imgfinal = Image.new("L", ((width, height)))
        imgfinal.putdata(array)
        imgfinal.save('testfinal.png')
        print "Test image saved to testfinal.png"

    def load_images_into_tensor_array(self, image_location_tuple_list, dimension=DIMENSION):
        """
        Return two arrays, first is array of TF image records, second is corresponding labels
        """
        image_files = [image for image, _ in image_location_tuple_list]
        tf_queue = tf.train.string_input_producer(image_files)
        reader = tf.WholeFileReader()
        key, value = reader.read(tf_queue)

    
        out_image = tf.image.rgb_to_grayscale(tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(value), DIMENSION, DIMENSION))

        sess = tf.Session()
        sess.as_default()
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_array = []

        for i in range(len(image_files)):          
            curr_image = out_image.eval(session=sess)
            tf_array.append(curr_image)

        #print len(tf_array)
        #unpack array, scale pixels
        for i in tf_array:
            img = list()
            for p in i:               
                for x in p:
                    #print x 
                    img.append(float(x[0]))
            self.final_images.append(img)
       # print img
        coord.request_stop()
        coord.join(threads)

def main():
    p = Parser("/home/jwatson1/compbio/jeremy/DavinciCoders/Data/mogged_imgs", True, False)
    p.pixel_to_image(p.final_images[40], DIMENSION, DIMENSION)

if __name__ == '__main__':
    main()
