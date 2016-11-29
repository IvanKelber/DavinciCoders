#runner.py

#(img_dim,batch_size,iters)
import subprocess

img_dims =[32,64,128,512]
batch_sizes = [64,25,9]
iters=[1000,2000]

starting_string = "python draw.py --source_dir=../11_Expressionism\
 --read_attn=True --write_attn=True "
count = 0
for dim in img_dims:
	for size in batch_sizes:
		for i in iters:
			data_dir = "script_test/exp_%d_%d_%d" % (dim,size,i)
			subprocess.call(["mkdir",(data_dir)])
			execution_string = starting_string + \
			"--data_dir=%s --img_dims=%d\
			 --batch_sizes=%d --iters=%d" %(data_dir,dim,size,i)
			subprocess.call(execution_string.split(" "))
			print "Done drawing %d_%d_%d in location: %s" %(dim,size,i,data_dir)
			count+=1
			if(count%5==0):
				subprocess.call(["python","plotter.py","./script_test"])

subprocess.call(["python","plotter.py","./script_test"])