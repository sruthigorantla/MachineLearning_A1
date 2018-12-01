import sys
import numpy
import pickle

numpy.seterr(all='ignore')
 
def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2

def argmax(x):
    # print(x.shape)
    predictions = numpy.zeros(x.shape)
    for i in range(len(x)):
        predictions[i][numpy.argmax(x[i])] = 1
    return predictions
class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)
        
        cost = self.negative_log_likelihood()
        return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return softmax(numpy.dot(x, self.W) + self.b)
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

def createConfusionMatrix(predictions, test_labels):
    conf_mat = numpy.zeros((8,8),dtype=float)
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
        r = numpy.argmax(test_labels[i])
        c = numpy.argmax(predictions[i])
        conf_mat[r][c] += 1
    
    return conf_mat
def test_lr(learning_rate=0.001, n_epochs=500):
    # training data

    with open("dataset.pickle","rb") as handle:
        dataset = pickle.load(handle)
    x = dataset['train_data']
    y = dataset['train_labels']
    # x = numpy.array([[1,1,1,0,0,0],
    #                  [1,0,1,0,0,0],
    #                  [1,1,1,0,0,0],
    #                  [0,0,1,1,1,0],
    #                  [0,0,1,1,0,0],
    #                  [0,0,1,1,1,0]])
    # y = numpy.array([[1, 0],
    #                  [1, 0],
    #                  [1, 0],
    #                  [0, 1],
    #                  [0, 1],
    #                  [0, 1]])


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=len(x[0]), n_out=len(y[0]))

    # train
    for epoch in range(n_epochs):
        classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()

        print ('Training epoch ',epoch)
        # print(argmax(classifier.predict(x)))
        print('cost is ', cost)
        learning_rate *= 0.95
        if(epoch%50 == 0):
            # test
            x = dataset['test_data']
            predictions = argmax(classifier.predict(x))
            accuracy = (numpy.argmax(predictions,1) == numpy.argmax(dataset['test_labels'],1)).sum().astype(float) / len(predictions)
            print("Test Accuracy: ",accuracy)
    x = dataset['train_data']
    predictions = argmax(classifier.predict(x))
    accuracy = (numpy.argmax(predictions,1) == numpy.argmax(dataset['train_labels'],1)).sum().astype(float) / len(predictions)
    print("Train Accuracy: ",accuracy)
    x = dataset['test_data']
    predictions = argmax(classifier.predict(x))
    accuracy = (numpy.argmax(predictions,1) == numpy.argmax(dataset['test_labels'],1)).sum().astype(float) / len(predictions)
    print("Test Accuracy: ",accuracy)
    confusion_matrix = createConfusionMatrix( predictions, dataset['test_labels'])
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            print(int(confusion_matrix[i][j]),end=" & ")
        print("\\\\\n")


if __name__ == "__main__":
    test_lr()