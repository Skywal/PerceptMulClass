import sys # needed to pass argv into QApplication
import os
from PyQt5 import QtWidgets
import pyqtgraph as pg

from design import perceptron_multi_class_rem as design 

import graph
import database as db
import neural_manager as n_manager


class PerceptMultClassApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        
        super().__init__()
        self.setupUi(self) 
        
        self.init_vars() 
        self.init_extern() # initialise all variables from external modules

        self.set_up_inputs() # set up start value at active text inputs 

        self.event_holders() # bind events and event holders
        
    
    def init_vars(self):
        """ Initial setup of the global variables. """

        self.max_num_epoch = 30 
        self.batch_size = 10  # how many times data will change in one training epo before weights will be updated
        self.accuracy = 0  # actual error of data evaluation
        self.loss = 0  # loss on datata evaluation
        self.learn_rate = 0.01
        self.rand_weights = True

        self.optimal_weights = []  

    def init_extern(self):
        """ Initialize all variables from orther modules aka by reference. """

        self.database = db.Database() 
        
        self.setup_graph(10)  # 10 possible data classes

        self.net_manager = n_manager.NetManager()


    def set_up_inputs(self):
        """ Set text in input boxes at start"""
        self.max_epoch_inp.setText(str(self.max_num_epoch))
        self.batch_size_inp.setText(str(self.batch_size))
        self.learn_rate_inp.setText(str(self.learn_rate))

        self.loss_inp.setText("")
        self.accuracy_inp.setText("")

    def read_inputs(self):
        """ Read data from active inputs """
        
        # Text inputs 

        epo_txt_inp = self.max_epoch_inp.text()
        btc_txt_inp = self.batch_size_inp.text()
        lnr_txt_inp = self.learn_rate_inp.text()

        if epo_txt_inp:
            self.max_num_epoch = int(epo_txt_inp)
        else:
            self.max_num_epoch = 0

        if btc_txt_inp:
            self.batch_size = int(btc_txt_inp)
        else:
            self.batch_size = 0
        
        if lnr_txt_inp:
            self.learn_rate = float(lnr_txt_inp)
        else:
            self.learn_rate = 0.1
        
        # Check boxes

        self.rand_weights = self.rand_weights_check_b.isChecked() 
    
    def restart_all(self):
        """ Bring all current state to start state """

        self.state_out_widg.clear() 
        
        self.plot_clear()

        self.init_vars() # reset all global variables
        self.set_up_inputs() # reset data in active text inputs

    
    def event_holders(self):
        """ Bind events """

        self.select_file_button.clicked.connect(self.browse_folder_action) 
        self.load_file_button.clicked.connect(self.load_data_action)
        self.start_button.clicked.connect(self.start_action)

    def browse_folder_action(self):
        """Select file from local disc by button press"""

        self.file_inp.clear() # cleat the line before writing
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open file")[0]

        if file_name:
            self.file_inp.setText(str(file_name)) # write a string 
    
    def load_data_action(self):
        """ Open selected file and read the data on Load button press """
        
        self.restart_all() # before start new session set up all to start state

        self.database.read_conv_calc_csv(self.file_inp.text())
        
        self.state_out_widg.addItem(f"{self.database.get_items_count()} input vectors in {self.database.get_items_dimensions()}-dimensional space have been loaded")

        self.plot_data() 

    def start_action(self):
        """ Begin action on start button """
        self.read_inputs()

        self.net_manager.set_up_vars(self.learn_rate, self.max_num_epoch, self.rand_weights, self.batch_size, 
                                    self.database.get_items_dimensions(), len(self.database.get_class_list()))

        self.neural_sequence()

        self.text_out(self.optimal_weights)

        self.plot_line()


    def setup_graph(self, data_classes):
        """ Initial setup of the graphical output """

        self.graph_widg = graph.Graph(data_classes=data_classes)
        self.graph_layout = QtWidgets.QVBoxLayout(self.graph_w)  # creating layout inside an empty widget
        
        self.graph_layout.addWidget(self.graph_widg)  # add graph widget insige layout
        
    def plot_data(self):
        """Plot data on the graph widget"""
        
        for i in range(len(self.database.get_class_list())):
            coords_list = list(self.database.data_separate(self.database.get_converted_data(), i))
            
            self.graph_widg.plot_dots_single_class(sequence_num=i, x=self.database.slice_column(coords_list), y=self.database.slice_column(coords_list, 1))
             
    def plot_clear(self):
        """ Replot all old data on zeros. """

        for i in range(self.graph_widg.get_data_cls()):
            self.graph_widg.plot_dots_single_class(sequence_num=i, x=[0], y=[0])
            self.graph_widg.plot_line(sequence_num=i, x=[0], y=[0])

    def plot_line(self):
        """ Plot line graps. """

        line_x = self.database.get_test_x_arrange()
        
        for i in range(len(self.optimal_weights)):
            
            weights_slice = self.database.slice_column(self.optimal_weights, i)
            
            line_y = self.net_manager.calc_line(weights_slice, line_x)
            
            self.graph_widg.plot_line(i, line_x, line_y)


    def neural_sequence(self):
        """ Neuron initialuze, train, predict and plot predictions sequence """

        self.net_manager.model_train_evaluate(self.database.get_numpy_data(), self.database.get_numpy_classes())    

        self.accuracy = self.net_manager.get_accuracy()
        self.loss = self.net_manager.get_data_loss()

        self.optimal_weights = list(self.net_manager.get_model_weights_list())


    def text_out(self, weights=[[0]]):
        """ Output text-info into all available information outputs """
        str_acc = "{0:.2f}".format(self.accuracy*100)
        str_loss = "{0:.2f}".format(self.loss*100)
        str_acc_loss = "Model loss = "+str_loss+" %, model accuracy = "+str_acc+" %"

        self.accuracy_inp.setText(str_acc)
        self.loss_inp.setText(str_loss)

        self.state_out_widg.addItem(f"\nMaximum number of epochs = {self.max_num_epoch}, batch size = {self.batch_size}")
        self.state_out_widg.addItem(str_acc_loss)
        self.state_out_widg.addItem(f"Optimal weights were found for {len(self.database.get_class_list())} data classes:")
        
        for i in range(len(weights)):
            
            wei_slice = self.database.slice_column(weights, i)

            self.state_out_widg.addItem(f"For neuron = {i+1} optimal weights are:")
            for j in range(len(wei_slice)):
                self.state_out_widg.addItem(f"w{j}={wei_slice[j]}")


     
def main():

    app = QtWidgets.QApplication(sys.argv) # new instance of QApplication
    window = PerceptMultClassApp() # create object of PerceptronApp
    window.show() # show the window
    app.exec_() # start app

if __name__ == "__main__":
    main()