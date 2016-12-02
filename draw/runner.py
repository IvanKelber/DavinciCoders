#runner.py

#(img_dim,batch_size,iters)
import subprocess
import sys
import time

img_dims =[64,256]
batch_sizes = [64]
iters=[2000]
canvasses=[20]
source_dir = ""
dest_dir = ""
try:
	source_dir = sys.argv[1]
except IndexError:
	source_dir="../11_Expressionism"

try:
	dest_dir = sys.argv[2]
except IndexError:
	dest_dir="../tmp"

starting_string = "python draw.py --source_dir=%s\
 --read_attn=True --write_attn=True " % (source_dir)
count = 0
subprocess.call(["mkdir",(dest_dir)])

dir_prefix= source_dir[6:9].lower()
for dim in img_dims:
	for size in batch_sizes:
		for i in iters:
			for canvas in canvasses:
				data_dir = "%s/%s_%d_%d_%d_%d" % (dest_dir,dir_prefix,dim,size,i,canvas)
				subprocess.call(["mkdir",(data_dir)])
				print "MADE DIRECTORY: %s" % data_dir
				execution_string = starting_string + \
				"--data_dir=%s --img_dim=%d\
				 --batch_size=%d --iters=%d\
				 --canvasses=%d" %(data_dir,dim,size,i,canvas)
				print "==== Populating Directory ==="
				start = time.clock()
				subprocess.call(execution_string.split(" "))
				end = time.clock()
				elapsed = abs(end-start)
				print "Done drawing %d_%d_%d_%d in location: %s" %(dim,size,i,canvas,data_dir)
				print "Elapsed Time: %d" % elapsed
				count+=1
				if(count%5==0):
					subprocess.call(["python","plotter.py",dest_dir])

subprocess.call(["python","plotter.py",dest_dir])