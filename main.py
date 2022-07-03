import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork():
    
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rating):
        self.i_nds = input_nodes
        self.h_nds = hidden_nodes
        self.o_nds = output_nodes

        self.lrn_rt = learning_rating

        self.wih = (numpy.random.normal(0.0 , pow(self.h_nds , -0.5) , (self.h_nds,self.i_nds)))
        self.who = (numpy.random.normal(0.0 , pow(self.o_nds , -0.5) , (self.o_nds,self.h_nds)))
        
        self.activation_func = lambda x: scipy.special.expit(x)


    def training(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        hidden_inputs =numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)


        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lrn_rt * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))

        self.wih += self.lrn_rt * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))



    def output(self,input_list):
        inputs = numpy.array(input_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih , inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rating = 0.3

data_file = open("training_set/mnist_train.csv","r")
data_list = data_file.readlines()
data_file.close()

ntw = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rating)

for record in data_list:

    all_values = record.split(",")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    
    ntw.training(inputs,targets)

test_data_file = open("training_set/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:

    all_values = record.split(",")
    correct = int(all_values[0])
    #print(int(all_values[0]),"is correct")

    inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    outputs = ntw.output(inputs)

    label_new = numpy.argmax(outputs)

    #print(label_new,"is what you guess")

    if correct == label_new:
        scorecard.append(1)
    else:
        scorecard.append(0)
#print(scorecard)   
score = scorecard.count(1)/len(scorecard)
print(score)