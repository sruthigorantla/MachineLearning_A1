import numpy as np
import pickle
import operator
vocab = {}
labels = []

count = 0
with open("./../r8-train-all-terms.txt","r") as fp:
# with open("testing_file.txt","r") as fp:
        for line in fp:
                count += 1
                lst = line.split("\t")
                words = lst[1].split()
                if( lst[0] not in labels ):
                        labels.append(lst[0])
                for word in words:
                        if( word in vocab ):
                                vocab[word] += 1
                        else:
                                vocab[word] = 1
print(count)
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

with open("./../r8-train-all-terms.txt","r") as fp:
        with open("train_processed.txt","w") as fp2:
                for x in sorted_vocab:
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
                                        line_dict[word] = 1
                        fp2.write(str(lst[0])+",")
                        for word in sorted_vocab:
                                if( word[0] in line_dict ):
                                        fp2.write(str(line_dict[word[0]])+",")
                                else:
                                        fp2.write("0,")
                        fp2.write("\n")
