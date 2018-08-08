
# coding: utf-8

# In[ ]:

import numpy
import scipy.special
#Neural network class def:

class neuralNetwork:
    
    #initialize the nn
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in input, hidden,outpou layers
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        
        #set learning rate
        self.lr=learningrate
        
        #link weight matrices with wih and who
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21 w12 w22 etc.
        #self.wih=(numpy.random.rand(self.hnodes,self.inodes)-0.5)
        #self.who=(numpy.random.rand(self.onodes,self.hnodes)-0.5)
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        #Activation function is the sigmoid function
        self.activation_function=lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        
        pass
    
    #train the neural net
    def train(self, inputs_list, target_lists):
        
        #convert input list to 2d array
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(target_lists,ndmin=2).T
        
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        
        #error
        output_errors=targets-final_outputs
        
        #hidden errors is the output errors split by weights, recombined at hidden nodes
        hidden_errors=numpy.dot(self.who.T,output_errors)
        
        #update the weights of links between the hidden and output layers
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #update the weights of links between the input and hidden layers
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        
        pass
    
    #query the neural net
    
    def query(self,inputs_list):
        #convert input list to 2d array
        inputs=numpy.array(inputs_list,ndmin=2).T
        
        
        #calculate the signals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        
        #calculates the signals emerging from the hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)
        
        
        #calculate the inputs of output layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        
        #calculate the result of output layer
        final_outputs=self.activation_function(final_inputs)
        
        return final_outputs


# In[ ]:


# backquery the neural network
    # we'll use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
