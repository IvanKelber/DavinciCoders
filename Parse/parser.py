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

        print len(self.images)

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

    def parse_jpg(jpg):
        return tf.image.decode_jpeg(jpg,channels=3,ratio=8)



    def load_images_into_tensor_array(image_location_tuple_list, dimension=DIMENSION):
        """
        Return two arrays, first is array of TF image records, second is corresponding labels
        """
        for img, location in image_location_tuple_list:



def main():
    p = Parser("Data/Pandora_18k/")


if __name__ == '__main__':
    main()
