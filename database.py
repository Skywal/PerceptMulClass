# Class to represent and work with training data.

import csv


class Database(object):
    
    def __init__(self):
        
        self.file_data = None   # raw data
        self.c_file_data = None  # converted into numbers data
        self.data_classes = None # list of unic classes in file


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
        """ Read data convert it into float and get all unic classes of values in the file. """

        self.read_convert_csv(file_path=file_path)
        
        self.data_classes = list(self.unic_classes())


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
        """ Cut entire n column from the n dimensional list into separate list. Work only for n x m list with equal row length. """

        L=[]
        
        if column <= 0:
            for i in input_list:
            # cut column in row and get value of that list item
                L.append(i[column][0])  #  convert from n x 1 list to 1 x n
            
            return L

        elif column >= len(input_list[0]):
            column = len(input_list[0])

        for i in input_list:
            L.append(i[(column - 1) : column][0])  
        
        return L


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
        """ Get values dimensions aka [0, 1, ..., n] """
        return len(self.file_data[0][:-1])
    
    def get_class_list(self):
        """ Get unic data classes. """
        return self.data_classes


if __name__ == "__main__":
    db = Database()
    #db.read_csv("D:/PROJECTS/LABKI/PerceptMulClass/example/sample1.csv")
    db.read_conv_calc_csv("D:/PROJECTS/LABKI/PerceptMulClass/example/sample1.csv")
    data_sep = db.data_separate(db.get_converted_data(), 0)
    
    print(db.slice_column(db.get_converted_data(), column=-1))
    #db.print_csv()