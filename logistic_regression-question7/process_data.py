import numpy as np
import pickle 
data = []
labels = []
mean = []
sigma = []
with open("documentation.txt","r") as handle:
	for line in handle:
		lst = line.split()
		mean.append(float(lst[3]))
		sigma.append(float(lst[4]))
def normalise(vector):
	vector_prime = [0]*len(vector)
	for i in range(len(vector)):
		vector_prime[i] = (vector[i] - mean[i])/sigma[i]
	return vector_prime
with open("./../spambase.data","r") as fp:
	for line in fp:
		lst = line.split(",")
		labels.append(int(lst[len(lst)-1][0]))
		vector = [float(x) for x in lst[0:-1]]
		data.append(normalise(vector))

data = np.asarray(data)
shuffled_indices = np.random.permutation(len(data))
print(shuffled_indices)
data_shuffled = []
data_shuffled = [data[i] for i in shuffled_indices]
labels_shuffled = [labels[i] for i in shuffled_indices]
#print(data.shape)
#print(labels)
#print(data[0])
data_shuffled = np.asarray(data_shuffled)
labels_shuffled = np.asarray(labels_shuffled)
split_index = int(0.8*len(data_shuffled))
print(split_index)
train_data = data_shuffled[:split_index]
test_data = data_shuffled[split_index:]
train_labels = labels_shuffled[:split_index]
test_labels = labels_shuffled[split_index:]
print(train_data[1])
print(len(train_data))
print(len(test_data))
print(len(data))
dataset = {}
dataset['train_data'] = train_data
dataset['train_labels'] = train_labels
dataset['test_data'] = test_data
dataset['test_labels'] = test_labels
with open("dataset.pickle","wb") as handle:
	pickle.dump(dataset, handle)

with open("train_data.txt","w") as fp:
	for i in range(len(train_data)):
		for j in range(len(train_data[i])):
			fp.write(str(train_data[i][j])+" ")
		fp.write("\n")
with open("train_labels.txt","w") as fp:
	for i in range(len(train_labels)):
		fp.write(str(train_labels[i])+"\n")
			
with open("test_data.txt","w") as fp:
	for i in range(len(test_data)):
		for j in range(len(test_data[i])):
			fp.write(str(test_data[i][j])+" ")
		fp.write("\n")
with open("test_labels.txt","w") as fp:
	for i in range(len(test_labels)):
		fp.write(str(test_labels[i])+"\n")