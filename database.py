# Class representing training data.

import csv


class Database(object):
    
    def __init__(self):
        
        self.file_data = None   # raw data
        self.data_classes = 0


    def read_csv(self, file_path):
        """ Read Comma Separated Values file from the local disk. """

        with open(file_path, mode='r') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            self.file_data = list(csv_reader) 

    def print_csv(self):
        """ Print readed data into screen line-by-line. """

        for row in self.file_data:
            print(f'\t{row}')
    
    def data_class_num(self):

        L = list(self.extract_class_l())

    def extract_class_l(self):
        """ Get list of classes from the file data. """

        L = [i[-1] for i in self.file_data]

        return L

    def data_separation(self, input_list=[], key_separator = 0):
        """ 
        Split input list by value in the last column
        input_list - list with data readet from the file in format [coord, coord, ..., class]
        key_separator - value in the last column of the list by whitch list can be splitted
        returns 
            list_cond - class value in the list is key_separator
            list_else - class value in the list is different from key_separator
        """
        list_cond = list()
        list_else = list()

        for row in input_list:
            if int(row[-1]) == key_separator:
                list_cond.append(row)
            else:
                list_else.append(row)
        
        return list_cond, list_else

    def slice_x(self, input_list):
        """ Cut entire first column from the two dimensional list into separate list """
        return[input_list[i][0:1] for i in range(len(input_list))]

    def slice_y(self, input):
        """ Cut entire second column from the two dimensional list into separate list """
        return[input_list[i][1:2] for i in range(len(input_list))]

    def get_data(self):
        """" Get raw data from the file in form of list"""
        return self.file_data
    
    def get_items_count(self):
        return len(self.file_data)
    
    def get_items_dimensions(self):
        return len(self.file_data[0][:-1])


if __name__ == "__main__":
    db = Database()
    db.read_csv("D:/PROJECTS/LABKI/PerceptMulClass/example/sample1.csv")
    #db.print_csv()
    
    