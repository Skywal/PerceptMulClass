from PyQt5 import Qt
import pyqtgraph as pg
import numpy as np



""" Widget class that displays graph """


class Graph(Qt.QWidget):

    def __init__(self):
        super().__init__()

        layout = Qt.QVBoxLayout(self) # create layout

        self.view = view = pg.PlotWidget() # make a plot area
        view.setBackground('w')
        
        # remove axis from plot
        view.hideAxis('left')
        view.hideAxis('bottom')
        
        layout.addWidget(self.view) # show up plot area

        self.line_pen = pg.mkPen(color='m', width=1.5)
        self.trand_line = view.plot(pen=self.line_pen) # create instance of a line

        # create instance of one of dots they will be reusable
        self.first_dot_cloud = view.plot(pen=None, symbol='o', symbolPen=None,
                                            symbolSize=5, symbolBrush='b')

        self.second_dot_cloud = view.plot(pen=None, symbol='t', symbolPen=None,
                                            symbolSize=5, symbolBrush='r')
        
    def clear_plot(self):
        """ Erase all from the plot, also deleting all displaying objects """
        self.view.clear()

    def plot_first_dots(self,x=[0], y=[0]):
        """ Draw dot like graph with dots coord 'x' and 'y' """
        self.first_dot_cloud.setData(x, y)
    
    def plot_second_dots(self,x=[0], y=[0]):
        """ Draw dot like graph with dots coord 'x' and 'y' """
        self.second_dot_cloud.setData(x, y)

    def plot_first_from_list(self, input_list):
        """ Plot blue dots """
        list_x, list_y = self.separate_coords(input_list)

        self.plot_first_dots(list_x, list_y)
    
    def plot_second_from_list(self, input_list):
        """ Plot red dots """
        list_x, list_y = self.separate_coords(input_list)

        self.plot_second_dots(list_x, list_y)
    
    def separate_coords(self, input_list):
        """ Split coords list into two separate lists """
        list_x = []
        list_y = []
        for row in input_list:
            list_x.append(float(row[0]))
            list_y.append(float(row[1]))

        return list_x, list_y

    def plot_line(self, x=[0], y=[0]):
        """Draw line graph with dots coord 'x' and 'y'"""
        self.trand_line.setData(x, y)

if __name__ == "__main__":
    app = Qt.QApplication([])
    w = Graph()
    w.show()
    app.exec()
