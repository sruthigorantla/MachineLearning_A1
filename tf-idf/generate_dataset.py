import numpy 
import pickle
import warnings
warnings.simplefilter("error", RuntimeWarning)
def normalise(vector,maximum,minimum):
    vector_prime = [0]*len(vector)
    for i in range(len(vector)):
        try:
            vector_prime[i] = (vector[i] - minimum[i])/(maximum[i] - minimum[i])
        except RuntimeWarning:
            vector_prime[i] = vector[i]
    return vector_prime
def createDataset():
    train_data = []
    train_labels = []
    label_dict = {}
    label_dict['earn'] = 0
    label_dict['money-fx'] = 1
    label_dict['trade'] = 2
    label_dict['acq'] = 3
    label_dict['grain'] = 4
    label_dict['interest'] = 5
    label_dict['crude'] = 6
    label_dict['ship'] = 7
    with open("train_processed.txt","r") as fp:
        count = 0
        for line in fp:
            if( count == 0 ):
                count += 1
                continue
            else:
                lst = line.split(",")
                train_labels.append(label_dict[lst[0]])
                train_data.append([float(x) for x in lst[1:-1]])
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
                test_labels.append(label_dict[lst[0]])
                test_data.append([float(x) for x in lst[1:-1]])
                count += 1
    print(len(train_data),len(test_data),len(train_labels),len(test_labels))
    return train_data, train_labels, test_data, test_labels

 
mean = []
sigma = []
train_data = []
test_data = []
data_train, labels_train, data_test, labels_test = createDataset()

minimum = numpy.amin(data_train,0)
maximum = numpy.amax(data_train,0)

print(minimum, maximum)

data_train = numpy.asarray(data_train)
for line in data_train:
    train_data.append(normalise(line,maximum,minimum))

data_test = numpy.asarray(data_test)
for line in data_test:
    test_data.append(normalise(line,maximum,minimum))
train_data = numpy.asarray(train_data)
test_data = numpy.asarray(test_data)

train_labels = numpy.zeros((len(data_train),8))
for i in range(len(labels_train)):
    train_labels[i][labels_train[i]] = 1

test_labels = numpy.zeros((len(data_test),8))
for i in range(len(labels_test)):
    test_labels[i][labels_test[i]] = 1
dataset = {}
dataset['train_data'] = train_data
dataset['train_labels'] = train_labels
dataset['test_data'] = test_data
dataset['test_labels'] = test_labels
with open("dataset.pickle","wb") as handle:
    pickle.dump(dataset,handle)