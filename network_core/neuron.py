""" Bias synapse located the end of the synapses sequence """
class Neuron(object):

    def __init__(self, synapses=1, stat_weight=0, learning_rate=1.0, max_error=.0):

        # initialize a list of synapses with weight 0 in count of synapses var
        # last synapse in the list is bias (always presents)
        self.synapses = [Synapse(start_weight=stat_weight) for i in range(synapses+1)]

        self.learning_rate = learning_rate # maximal learning rate 
        self.max_error = max_error # maximal error at epoch

        self.adder_res = 0 # result of adding weight-multiply-data process
        self.activation_res = 0 # result of activation function
        self.curr_error = 0 # error at current learning epoch


    def get_synapses(self):
        return self.synapses
    
    def get_weights(self):
        """ Get list of all synapses weights for current state of the neuron """
        return [i.get_weight() for i in self.synapses]
        
    def adder(self, input_list=[0]):
        """ Adding all rezults of synapses work plus bias"""
        summ = 0

        '''
        for syn, data in zip(self.synapses, input_list):
            summ = summ + syn.think_process(data)
        '''
        for i in range(len(input_list)):
            summ = summ + self.synapses[i].think_process(input_list[i])

        # bias neuron
        summ = summ + self.synapses[-1].think_process(1)
       
        return summ

    def adder_process(self, input_list=[0]):
        """ Summ of all synapse * input_data """
        self.adder_res = self.adder(input_list=input_list)

    def heviside_activation(self):
        """ Heviside function implementation """

        if self.adder_res >= 0:
            self.activation_res = 1
        else:
            self.activation_res = 0
    

    def fit_bias(self, out_sig):
        """ Adjusting bias weight"""
        result = self.synapses[-1].get_weight() + (self.learning_rate * (out_sig - self.activation_res))
        return result

    def fit_weights(self, input_data, output_signal):
        """ 
        input_data - list of data for all synapses 
        output_signal - corresponding signal for input data vector  aka d (last column in the local file data 'example*.csv')
        """
        # last one is bias and it's calculated in separate way
        for i in range(len(self.synapses)-1):
            old_w = self.synapses[i].get_weight()
            new_w = old_w + (self.learning_rate * input_data[i] * (output_signal - self.activation_res))

            self.synapses[i].set_new_weight(new_w)
        
        # calculate bias weight
        bias = self.fit_bias(output_signal)
        self.synapses[-1].set_new_weight(bias)
    """
    def train_network(self, train_data, epochs=59):
        curr_ep = 0
        
        while curr_ep < epochs:
            
            self.curr_error = 0

            for i in range(len(train_data)):
                data_list = list(train_data[i][:-1])
                data_list = [float(i) for i in data_list]

                output_signal = int(train_data[i][-1])

                self.adder_process(data_list)
                self.heviside_activation()
                
                self.fit_weights(input_data=data_list, output_signal=output_signal)

                self.error_calc(output_signal)
            
            print(f"Current error is {self.curr_error}")
            
            if self.curr_error <= self.max_error:
                print(f"Optimal weights were found after {curr_ep} epochs.")
                break
            
            curr_ep = curr_ep + 1
            
        print(f"Epochs {curr_ep}")
    """

    def error_calc(self, out_sig):
        self.curr_error = self.curr_error + abs(out_sig - self.activation_res)


class Synapse(object):

    def __init__(self, start_weight=0):

        self.weight = start_weight
    
    def set_new_weight(self, weight):
        self.weight = weight
    
    def get_weight(self):
        return self.weight

    def think_process(self, input_data):
        return (self.weight * input_data)


if __name__ == "__main__":
    neural = Neuron(synapses=2, max_error=1)

    database = db.Database()
    database.read_csv("D:/PROJECTS/LABKI/Perceptron/example/sample2.csv")

    neural.train_network(database.get_data(), epochs=58)
    
    ''''
    data = list(database.get_data())
    data_list = []
    output_sig = []
    for i in range(len(data)):
        data_list = list(data[i][:-1])
        output_sig = int(data[i][-1])
        data_list = [float(i) for i in data_list]
        print(f"{i} Data list is {data_list}    and out signal is {output_sig}")
    '''

    for i in neural.get_synapses():
        print(f"neuron num has weight {i.get_weight()}")
