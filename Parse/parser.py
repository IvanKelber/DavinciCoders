import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join


DIMENSION = 64

class Parser:

    def __init__(self,path):

        print self.get_images(path)


    def get_images(self,era):
        artists = [artist for artist in os.listdir(era) if not isfile(join(era,artist))]
        images = []
        for artist in artists:
            images += [(image,era) for image in os.listdir(join(era,artist))]
        return images
    def parse_image(image):
        pass

def main():
    p = Parser("Data/Pandora_18k/05_Baroque")


if __name__ == '__main__':
    main()
