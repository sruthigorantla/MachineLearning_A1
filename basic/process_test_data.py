import numpy as np
import pickle
vocab = []
labels = []

with open("vocab.pickle","rb") as handle:
	vocab = pickle.load(handle)
print(len(vocab))
with open("./../r8-test-all-terms.txt","r") as fp:
	with open("test_processed.txt","w") as fp2:
		for x in vocab:
			fp2.write(x[0]+",") 
		fp2.write("\n")
		for line in fp:
			lst = line.split("\t")
			words = lst[1].split()
			line_dict = {}
			for word in words:
				if( word in line_dict ):
					line_dict[word] += 1
				else: 
					line_dict[word] = 0
			fp2.write(str(lst[0])+",")
			for word in vocab:
				if( word[0] in line_dict ):
					fp2.write(str(line_dict[word[0]])+",")
				else:
					fp2.write("0,")
			fp2.write("\n")

