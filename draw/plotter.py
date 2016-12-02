#plotter.py

import sys
import subprocess
import os

root = sys.argv[1]

os.chdir(root)
for directory in os.listdir(os.getcwd()):
	if(not os.path.isfile(directory)):
		os.chdir(directory)
		plot_string = "python ../../plot_data.py mogged draw_data.npy"
		subprocess.call(plot_string.split(" "))
		os.chdir("..")



