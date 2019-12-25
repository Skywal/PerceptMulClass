from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.utils import plot_model
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np

import database as db


"""Managing neural network work"""
class NetManager(object):
    def __init__(self, inp_dims=2, epochs=100, learning_rate=0.001, batch_size=500, rand_weights=True, data_classes=3):
        
        self.init_vars()
        self.set_up_vars(lnr_rate=learning_rate, epo=epochs, rand_weights=rand_weights, 
                        btc_size=batch_size, inp_dims=inp_dims, d_classes=data_classes)

        
    def init_vars(self, ):
        """ Initialise global variables """
        self.model = None

        self.mod_loss = 0  #  model loss based on evaluation
        self.mod_accuracy = 0  # model accuracy based on evaluation (*100 to get percents)
        self.mod_predict_acc = 0  # model accuracy on predictions
        self.mod_weights = []  # list of all model weights

        self.dummy_y = 0
        self.X_train = 0
        self.X_test = 0
        self.Y_train = 0
        self.Y_test = 0

        self.test_size = .2  # percent of all data that will be transfered into test list 
    
    def set_up_vars(self, lnr_rate, epo, rand_weights, btc_size, inp_dims, d_classes):
        
        self.batch_size = btc_size
        self.epochs = epo
        self.lnr_rate = lnr_rate
        self.rand_weights = rand_weights
        self.init_inp_dim = inp_dims
        self.data_classes = d_classes

    def init_model(self):
        """ Initialise Adam optimizer and create and compile keras Sequential model. """
        self.adam_opt = optimizers.Adam(learning_rate=self.lnr_rate)

        self.model = self.form_my_model() 

    
    def rand_init_weights(self):
        """ Set up initial weights to random or zeros depending on 'self.rand_weights' variable value. 'True' - random values, 'False' - zeros """

        if self.rand_weights:
            return 'random_uniform'
        else:
            return 'zeros'

    def form_my_model(self):
        """ Create and return keras Sequential model with 1 Dense layers, with activation 'sigmoid'.
        Number neurons depends on 'data_classes' variable and input depends on 'init_inp_dim'.
        Loss function - 'categorical_crossentropy', optimizer is 'adam_opt' and mertics are 'categorical_accuracy'. """
        
        model = Sequential()
        model.add(Dense(units=self.data_classes, input_dim=self.init_inp_dim, 
                        activation='sigmoid',
                        bias_initializer=self.rand_init_weights(), 
                        kernel_initializer=self.rand_init_weights()))
        
        #model.add(Dropout(0.01))  # helps to deal with overfit

        model.compile(loss='categorical_crossentropy', optimizer=self.adam_opt, metrics=['categorical_accuracy'])

        return model


    def make_dummy_y(self, class_list):
        # because it's a multiclass problem labels should be transformed into the categorical format
        self.dummy_y = np_utils.to_categorical(class_list)

    def make_train_test_data(self, data_list, class_list):
        """ Split and prepare input data. Size of '_test' set depends on 'test_size' in percents. """
        self.make_dummy_y(class_list)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(data_list, self.dummy_y, test_size=self.test_size)


    def model_train_evaluate(self, data_list, class_list):
        """ Train and evaluate model. """
        self.init_model()
        
        self.make_train_test_data(data_list, class_list)

        self.model.fit(x=self.X_train, y=self.Y_train, 
                       batch_size=self.batch_size, 
                       epochs=self.epochs, 
                       shuffle=False,
                       validation_data=(self.X_test, self.Y_test), 
                       verbose=1)

        self.mod_loss, self.mod_accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        self.predict_and_estimate()
        
        self.model.predict_proba(self.X_test)

        plot_model(self.model, to_file="model.png")  # plot model structure
                
    def predict_and_estimate(self):
        """ Make prediction on '_test' set and estimate prediction accuracy. """
        self.prediction = self.model.predict_proba(self.X_test)
        self.mod_predict_acc = roc_auc_score(y_true=self.Y_test, y_score=self.prediction)

    
    def calc_line(self, weights=[0], data=[0]):
        """ Calculate dots of trand line for 'data' 1-dim list of dots. 'weights' is a list of model weights for data class of 'data' list. Return list."""
        a = round(float(weights[1]), 4)  # weight for input x_1
        b = round(float(weights[2]), 4)  # weight for input x_2
        c = round(float(weights[0]), 4)  # bias weight

        result = []
        
        for i in data:
            equation = (-1.0)*((a * float(i)) + c) / (b)
            result.append(equation)
        return result


    def get_class_prediction(self, x):
        """ Get single set of data class prediction. """
        return self.model.predict_classes(x)

    def get_prediction(self, x):
        """ Get prediction for a list. """
        return self.model.predict(x)

    def get_model_weights(self):
        """ Get weights from the model layers. First is kernel matrix, last are biases. Columns are Neurons and rows are inputs, last vector is bias. All data is numpy array. """ 
        for layer in self.model.layers:
            weights = layer.get_weights()
        return weights

    def get_model_weights_list(self):
        """ Get all weights from model in form of single list [[w0_1, w0_2, ..., w0_n] [w1_1, w1_2, ..., w1_n] [wn_1, wn_2, ..., wn_n]] """
        weig = self.get_model_weights()
               
        result = []
        
        result.append(weig[1].tolist())  # get bias weights 
        # get all other weights 
        for i in weig[0]:
            result.append(i.tolist())

        return result

    def get_accuracy(self):
        """ Model prediction accuracy. """
        return self.mod_predict_acc
    
    def get_data_loss(self):
        """ Model data loss on dataset evaluation. """
        return self.mod_loss
    

if __name__ == "__main__":
    # test

    database = db.Database()
    database.read_conv_calc_csv("D:/PROJECTS/LABKI/PerceptMulClass/example/sample1.csv")
    
    csv_list = database.get_numpy_all_csv()
    print(csv_list)
    #print(database.get_items_dimensions())
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    data_list = database.get_numpy_data()
    class_list = database.get_numpy_classes()

    net_manager = NetManager(inp_dims=database.get_items_dimensions(), epochs=30, batch_size=10, rand_weights=True)

    net_manager.model_train_evaluate(data_list, class_list)
    
    print("\nModel weights after")
    #wei_after = net_manager.get_model_weights()
    #for i in wei_after:
    #    print(i)
    #print("\n")
    print("++++++++++++++++++++")
    #print(net_manager.get_l0_weights())
    print("------------------------")
    #print(net_manager.get_l0_bias())
    print("\n")

    arr = [[-4.438424802704748, -6.5223773600741834], ## 0
     [-3.085393136916536,-4.820753879421525], ## 0
     [11.102347458781438,1.4715952203420422], ## 1
     [6.205164330259273,-13.453352092582335]] ## 2
    pred_data = np.asarray([[6.205164330259273,-13.453352092582335]], dtype=np.float32)
    print(pred_data)
    predictions = net_manager.get_class_prediction(pred_data)
    #predictions = net_manager.get_prediction(pred_data)
    print(predictions)
    print("***************************************")

    for i in net_manager.get_model_weights_list():
        print(i)

    print("++++++++++")
    print(net_manager.get_model_weights())