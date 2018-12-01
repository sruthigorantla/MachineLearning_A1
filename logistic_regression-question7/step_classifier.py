import numpy as np
import math
import pickle
from sklearn.linear_model import LogisticRegression
def sigmoid(scores):
	return 1 / (1 + np.exp(-scores))

def log_likelihood( features, target, weights ):
	scores = np.dot( features, weights )
	ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
	return ll

def logistic_regression( features, target, num_steps, learning_rate, add_intercept = False ):
	if add_intercept:
		intercept = np.ones((features.shape[0], 1))
		features = np.hstack((intercept, features))
	weights = np.zeros(features.shape[1])
	for step in range(num_steps):
		scores = np.dot( features, weights )
		#print(scores)
		predictions = sigmoid(scores)
		#print(predictions)
		output_error_signal = target - predictions
		#print(output_error_signal)
		gradient = np.dot(features.T, output_error_signal)
		#print(gradient)
		weights += learning_rate*gradient
		#print(weights)
		# if step % 10000 == 0:
		# 	print(log_likelihood( features, target, weights ))
	return weights

def train(features, labels):
	weights = logistic_regression( features, labels, num_steps = 300000, learning_rate = 5e-5, add_intercept = True )
	return weights

	
# with open("dataset.pickle","rb") as handle:
# 	dataset = pickle.load(handle)
dataset = {}
dataset['train_data'] = []
dataset['train_labels'] = []
dataset['test_data'] = []
dataset['test_labels'] = []
with open("train_data.txt","r") as fp:
	for line in fp:
		lst = line.split(" ")
		dataset['train_data'].append([float(x) for x in lst[:-1]])
with open("test_data.txt","r") as fp:
	for line in fp:
		lst = line.split(" ")
		dataset['test_data'].append([float(x) for x in lst[:-1]])
with open("train_labels.txt","r") as fp:
	for line in fp:
		dataset['train_labels'].append(int(line[0]))
with open("test_labels.txt","r") as fp:
	for line in fp:
		dataset['test_labels'].append(int(line[0]))

length = len(dataset['train_data'])
for i in range(10,110,10):	
	print("i=",i)
	l = int(float(i*length)/100)
	features = np.asarray(dataset['train_data'][0:l])
	labels = np.asarray(dataset['train_labels'][0:l])
	test_features = np.asarray(dataset['test_data'])
	test_labels = np.asarray(dataset['test_labels'])
	weights = train(features, labels)
	data_with_intercept = np.hstack((np.ones((features.shape[0], 1)), features))
	final_scores = np.dot(data_with_intercept, weights)
	predictions = np.round(sigmoid(final_scores))
	train_accuracy = (predictions == labels).sum().astype(float) / len(predictions)
	print("Train Accuracy: ",train_accuracy)
	print("Train error: ",1-train_accuracy)
	data_with_intercept = np.hstack((np.ones((test_features.shape[0], 1)), test_features))
	final_scores = np.dot(data_with_intercept, weights)
	predictions = np.round(sigmoid(final_scores))
	test_accuracy = (predictions == test_labels).sum().astype(float) / len(predictions)
	print("Test Accuracy: ",test_accuracy)
	print("Test Error: ",1-test_accuracy)

# clf = LogisticRegression(fit_intercept=True, C = 1e15)
# clf.fit(features, labels)

# print(clf.intercept_, clf.coef_)
bias = weights[0]
weights = weights[1:]
print(weights)
print(bias)
# print("acc form sklearn ",clf.score(test_features, test_labels))
