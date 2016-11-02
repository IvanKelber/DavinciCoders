import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join


DIMENSION = 64

class Parser:

    def __init__(self,path):
        self.images = []
        for era in os.listdir(path):
            if(not isfile(era)):
                self.images += self.get_images(join(path,era))
        print self.images
        self.load_images_into_tensor_array(self.images)

    def get_images(self,era):
        """
        Returns all of the images associated with that era in a list of tuples of form:
        (image_filename,path_to_era)
        """
        artists = [artist for artist in os.listdir(era) if not isfile(join(era,artist))]
        images = []
        for artist in artists:
            images += [(image,era) for image in os.listdir(join(era,artist))]
        return images

    def parse_image(image):
        pass

    def load_images_into_tensor_array(self, image_location_tuple_list, dimension=DIMENSION):
        """
        Return two arrays, first is array of TF image records, second is corresponding labels
        """
        image_files = [join(image_path,image) for image, image_path in image_location_tuple_list]
        tf_queue = tf.train.string_input_producer(image_files)
        reader = tf.WholeFileReader()
        key, value = reader.read(tf_queue)

        out_image = tf.image.decode_jpeg(value)

        sess = tf.Session()
        sess.as_default()
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_array = []

        for i in range(len(image_files)):
            curr_image = out_image.eval(session=sess)
            tf_array.append(curr_image)

        print len(tf_array)

        coord.request_stop()
        coord.join(threads)




def main():
    p = Parser("Data/Pandora_18k/")


if __name__ == '__main__':
    main()