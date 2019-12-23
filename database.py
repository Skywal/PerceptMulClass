# Class to represent and work training data.

import numpy as np
import csv


class Database(object):
    
    def __init__(self):
        
        self.file_data = None   # raw data
        self.c_file_data = None  # converted into numbers data
        self.data_classes = None # list of unic classes in file
        
        self.numpy_all_cvs = None
        self.numpy_data = None
        self.numpy_classes = None


    def read_csv(self, file_path):
        """ Read Comma Separated Values file from the local disk. """

        with open(file_path, mode='r') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            self.file_data = list(csv_reader) 
    
    def read_convert_csv(self, file_path):
        """ Read data and convert it into float. Converted data is in c_file_data list. """

        self.read_csv(file_path=file_path)
        self.convert_f_data()

    def read_conv_calc_csv(self, file_path):
        """ Read data convert it into float and get all unic classes of values in the file. Also transfer all into numpy arrays. """

        self.read_convert_csv(file_path=file_path)
        
        self.data_classes = list(self.unic_classes())

        self.convert_csv_to_numpy()
        self.convert_data_class_to_numpy()


    def print_csv(self):
        """ Print readed data into screen line-by-line. """

        for row in self.file_data:
            print(f'\t{row}')
    
    def convert_f_data(self):
        """ Converting data form file from strings into float. Converted data appending to c_file_data list. """
        
        self.c_file_data = []

        for i in self.file_data:
            self.c_file_data.append([float(j) for j in i])

    def unic_classes(self):
        """ Get list of all unic classes in data file. """

        L = list(self.extract_class_line(self.c_file_data))
        
        unic_nums = []

        for i in L:
            if i not in unic_nums:
                unic_nums.append(i)
        
        return unic_nums

    def extract_class_line(self, file_data):
        """ Get list of classes from the file data (last column). Returns a list. """

        L = [i[-1] for i in file_data]

        return L

    def data_separate(self, input_list=[], key_separator = 0):
        """ 
        Split input list by value in the last column
        input_list - list with data readed from the file in format [coord, coord, ..., class]
        key_separator - value in the last column of the list by whitch list can be splitted
        returns 
            a list with all values of the class that fits to key_separator
        """
        result = list()
    
        for row in input_list:
            if int(row[-1]) == key_separator:
                result.append(row)
        
        return result

    def slice_column(self, input_list, column=0):
        """ Cut entire n column from the n dimensional list into separate list. Work only for n x m list with equal row length. 
        column - upped border in slice which will not be taken, so to slice second column from list it should be set 2"""

        L=[]
        
        if column <= 0:
            for i in input_list:
            # cut column in row and get value of that list item
                L.append(i[column])  
            
            return L

        elif column >= len(input_list[0]):
            column = len(input_list[0])

        for i in input_list:
            L.append(i[(column - 1) : column][0])  #  convert from 1 x n list to n x 1
        
        return L
    
    def split_data_and_class(self):
        """ Slice data and class into separate lists. Return d_list, c_list. """
        d_list = []  # data
        c_list = []  # classes
        
        for i in self.c_file_data:
            d_list.append(i[:-1])
            c_list.append(i[-1])
        
        return d_list, c_list

    
    def get_data(self):
        """ Get raw data from the file in form of list. """
        return self.file_data
    
    def get_converted_data(self):
        """ Get data transformed into float. """
        return self.c_file_data

    def get_items_count(self):
        """ All amount of data lines in the file. """
        return len(self.file_data)
    
    def get_items_dimensions(self):
        """ Get values dimensions aka [0, 1, ..., n] in form of single num 'n'. """
        return len(self.file_data[0][:-1])
    
    def get_class_list(self):
        """ Get unic data classes. """
        return self.data_classes


    def convert_csv_to_numpy(self):
        self.numpy_all_cvs = np.asarray(self.c_file_data, dtype=np.float32)
    
    def convert_data_class_to_numpy(self):

        d_list, c_list = self.split_data_and_class()

        self.numpy_data = np.asarray(d_list, dtype=np.float32)
        self.numpy_classes = np.asarray(c_list, dtype=np.float32)

    def get_numpy_all_csv(self):
        return self.numpy_all_cvs

    def get_numpy_data(self):
        return self.numpy_data

    def get_numpy_classes(self):
        return self.numpy_classes


if __name__ == "__main__":
    # Tests

    db = Database()
    #db.read_csv("D:/PROJECTS/LABKI/PerceptMulClass/example/sample1.csv")
    db.read_conv_calc_csv("D:/PROJECTS/LABKI/PerceptMulClass/example/sample1.csv")
    #data_sep = db.data_separate(db.get_converted_data(), 0)
    
    #print(db.get_converted_data())

    #print(db.slice_column(db.get_converted_data(), column=-1))
    
    list_zero = list(db.data_separate(db.get_converted_data()))
    list_one = list(db.data_separate(db.get_converted_data(), key_separator=1)) # split input data by last row value
    
    #print(list_zero)
    #print(list_one)
    
    #print(db.slice_column(list_zero))
    #print(db.slice_column(list_zero, 2))

    #print(db.slice_column(list_one))
    #print(db.slice_column(list_one, 1))
    
    #data, classes = db.split_data_and_class()
    #print(data)
    #print(classes)
    print(db.get_numpy_all_csv())
