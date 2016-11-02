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
        artists = [artist for artist in os.listdir(era) if not isfile(join(era,artist))]
        images = []
        for artist in artists:
            images += [(image,era) for image in os.listdir(join(era,artist))]
        return images
    def parse_image(image):
        pass

def main():
    p = Parser("Data/Pandora_18k/")


if __name__ == '__main__':
    main()
