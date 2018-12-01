import numpy as np
import math
import pickle
import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

vocab = {}
labels = []
D = 0
with open("./../r8-train-all-terms.txt","r") as fp:
	for line in fp:
		D += 1
		lst = line.split("\t")
		words = lst[1].split()
		words = [w for w in words if not w in stop_words]
		words = [ps.stem(w) for w in words]
		if( lst[0] not in labels ):
			labels.append(lst[0])
		for word in words:
			if( word in vocab ):
				vocab[word] += 1
			else:
				vocab[word] = 0
print(len(vocab))
new_vocab = {}
for k, v in vocab.items():
        new_k = k.lower()
        if( new_k in new_vocab ):
                print(v)
                new_vocab[new_k] += v
        else:
                new_vocab[new_k] = v
sorted_vocab = sorted(new_vocab.items(), key=operator.itemgetter(0))

with open("vocab.pickle","wb") as handle:
	pickle.dump(sorted_vocab, handle)

term_dict = {}
for k, v in vocab.items():
	term_dict[k] = 0.0
idf = {}
with open("./../r8-train-all-terms.txt","r") as fp:
	for line in fp:
		lst = line.split("\t")
		words = lst[1].split()
		for k, v in vocab.items():
			if( k in words ):
				term_dict[k] += 1		
for k, v in term_dict.items():
	idf[k] = math.log(float(D)/(1+v))

with open("idf.pickle","wb") as pp:
	pickle.dump(idf, pp)

with open("./../r8-train-all-terms.txt","r") as fp:
	with open("train_processed.txt","w") as fp2:
		for x in sorted_vocab:
			fp2.write(x[0]+",") 
		fp2.write("\n")
		for line in fp:
			lst = line.split("\t")
			words = lst[1].split()
			words = [w for w in words if not w in stop_words]
			words = [ps.stem(w) for w in words]
			line_dict = {}
			for word in words:
				if( word in line_dict ):
					line_dict[word] += 1
				else: 
					line_dict[word] = 0
			fp2.write(str(lst[0])+",")
			for word in vocab:
				if( word in line_dict ):
					fp2.write(str((line_dict[word])*(float(D)/idf[word]))+",")
				else:
					fp2.write("0,")
			fp2.write("\n")

