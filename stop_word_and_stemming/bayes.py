import math
import numpy as np
import pickle
def separate(train_data, train_labels):
	classes_dict = {}
	for i in range(len(train_data)):
		if( train_labels[i] not in classes_dict ):
			classes_dict[train_labels[i]] = []
		classes_dict[train_labels[i]].append(train_data[i])
	print(len(classes_dict))
	return classes_dict

def getModel(train_data, train_labels):
	train_separated = separate(train_data, train_labels)
	class_probability = {}
	for classname, examples in train_separated.items():
		class_probability[classname] = float(len(examples))/len(train_data)
	return train_separated, class_probability

def getVocab():
	with open("vocab.pickle","rb") as handle:
		vocab = pickle.load(handle)
	mod_V = len(vocab)
	return mod_V
	
def classInfo(examples):
	count = 0
	for i in range(len(examples)):
		for j in range(len(examples[i])):
			if( examples[i][j] > 0 ):	
				count += examples[i][j]
	return count
	
def Predict( train_separated, class_probability, test_data):
	mod_V = getVocab()
	predictions = []
	class_dict = {}
	class_sum = {}
	for k, v in train_separated.items():
		class_dict[k] = classInfo(v)
	for k, v in train_separated.items():
		class_sum[k] = np.asarray(v).sum(axis=0)
	for i in range(len(test_data)):
		example = test_data[i]
		max_class = ""
		max_prod = 0.0
		for c, lst in train_separated.items():
			prod = 1.0
			denominator = class_dict[k] + mod_V
			for i in range(len(example)):
				if( example[i] > 0):
					if(prod*float(class_sum[c][i]+1)/denominator == 0):
						continue
					prod *= float(class_sum[c][i]+1)/denominator
			if(prod*class_probability[c] > 0 ):
				prod *= class_probability[c]
			if( prod > max_prod ):
				max_prod = prod
				max_class = c
		predictions.append(max_class)
	return predictions

def getAccuracy( predictions, test_labels):
	count = 0
	for i in range(len(test_labels)):
		if(predictions[i] == test_labels[i]):
			count += 1
	return float(count)/len(test_labels)

def createDataset():
	train_data = []
	train_labels = []
	with open("train_processed.txt","r") as fp:
		count = 0
		for line in fp:
			if( count == 0 ):
				count += 1
				continue
			else:
				lst = line.split(",")
				train_labels.append(lst[0])
				train_data.append([int(x) for x in lst[1:-1]])
				count += 1
	test_data = []
	test_labels = []
	with open("test_processed.txt","r") as fp:
		count = 0
		for line in fp:
			if( count == 0 ):
				count += 1
				continue
			else:
				lst = line.split(",")
				test_labels.append(lst[0])
				test_data.append([int(x) for x in lst[1:-1]])
				count += 1
	print(len(train_data),len(test_data),len(train_labels),len(test_labels))
	return train_data, train_labels, test_data, test_labels

def createConfusionMatrix(train_separated, predictions, test_labels):
	conf_mat = np.zeros((8,8))
	label_dict = {}
	label_dict['earn'] = 0
	label_dict['money-fx'] = 1
	label_dict['trade'] = 2
	label_dict['acq'] = 3
	label_dict['grain'] = 4
	label_dict['interest'] = 5
	label_dict['crude'] = 6
	label_dict['ship'] = 7
	print(len(label_dict))
	print(label_dict)
	for i in range(len(test_labels)):
		r = label_dict[test_labels[i]]
		c = label_dict[predictions[i]]
		conf_mat[r][c] += 1
	
	return conf_mat

def main():	
	train_data, train_labels, test_data, test_labels = createDataset()
	train_separated, class_probability = getModel(train_data, train_labels)
	print(class_probability)
	predictions = Predict(train_separated, class_probability, test_data)
	print(predictions)
	
	accuracy = getAccuracy(predictions, test_labels)	
	print("Test Accuracy: ",accuracy)
	confusion_matrix = createConfusionMatrix(train_separated, predictions, test_labels)
	for i in range(len(confusion_matrix)):
		for j in range(len(confusion_matrix[i])):
			print(int(confusion_matrix[i][j]),end=" & ")
		print("\\\\\n")
	predictions = Predict(train_separated, class_probability, train_data)
	accuracy = getAccuracy(predictions, train_labels)
	print("Train Accuracy: ",accuracy)

main()