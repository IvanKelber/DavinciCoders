Members:
Josh Cusano
Ivan Kelber
Shreyas Sundararaman
Jeremy Watson
Ashley Conard

Idea:
There are several applications of deep learning to stylize a particular image.  Usually, the user provides an image, selects the style, and then their image is recreated according to that style.  Our goal is slightly different: given paintings of a particular style as input, generate an original painting of a similar style.  For example, if we train on a dataset of impressionist paintings, our net would generate a painting that would pass as an original impressionist painting.

Relevant Papers:

DeepDream
DRAW
Pixel Recurrent Neural Networks - https://arxiv.org/pdf/1601.06759v3

Dataset:
http://imag.pub.ro/pandora/pandora_download.html

Specific Data Information:

In each folder for type of art there are the various artists.  In each artist there
are the various paintings of that artist.  The only exception is "Abstract_Art"
which has subtypes of abstract art.  To make our lives easier just take the artists
from each subtype and move them to the main "Abstract_Art" folder.  Some artists
are in more than one subtype so you'll need to merge their paintings.

###DEPRECATED#####
Basic LSTM:
Takes an image, trains on 90%, tests on 10%, outputs an image based on test data.
Dependencies: (needs to be formalized)
Tensorflow
PIL
Numpy
Sys
To run:
python main.py <image-to-train> 

####CURRENT-BUILD#####
draw:
The author of this code is Eric Jang, https://github.com/ericjang/draw. We have made minor tweaks to work with our images.

Setup:
kinit
ssh cslab6a (or cslab6b or cslab6c)
source /course/cs2950k/tf_gpu_venv/bin/activate 

Run:
If you'd like to try a particular configuration of parameters or a particular dataset you can run
python draw.py --data_dir=./tmp/ --source_dir=./path/to/source/ --read_attn=True --write_attn=True --img_dim=64 --batch_size=100 --iters=1000 --canvasses=10

If you'd like to run several different combinations of parameters on a JUST expressionism then:
mkdir ./script_test
python runner.py

NOTE that this requires a directory of images.  Right now it's hard coded to use 11_Expressionism/ as a source.  It also names the generated folders containing the generated images with an "exp" prefix to acknowledge the type.  A small TODO: pass in source into runner.py and name the folders accordingly.

To construct generated images from the numpy binary:
python plot_data.py planes draw_data.npy

If you'd like to construct generated images from several different folders containing numpy binaries:
python plotter.py /path/to/directory

