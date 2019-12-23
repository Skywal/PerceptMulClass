#  Widget class that displays graph.

from PyQt5 import Qt
import numpy as np

import pyqtgraph as pg
import pyqtgraph.examples


class Graph(Qt.QWidget):

    def __init__(self, data_classes=1):
        super().__init__()

        layout = Qt.QVBoxLayout(self) # create layout

        self.view = self.init_plot()
        
        layout.addWidget(self.view) # show up plot area
        
        self.init_vars()

        self.init_dots_graphs(data_classes=data_classes)
        self.init_line_graphs(data_classes=data_classes)


    def init_plot(self):
        """ Initialize and set up plot widget. Return widget object. """

        view = pg.PlotWidget() # make a plot area
        
        view.setBackground((255, 255, 255, 0))
        view.setAntialiasing(True)
        view.setAspectLocked(True)
        view.enableMouse(False)
        # remove axis from plot
        view.hideAxis('left')
        view.hideAxis('bottom')
        
        return view
    
    def init_vars(self):

        self.plot_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', (9, 59, 198), (215, 145, 5)]
        self.dots_obj_list = []
        self.line_obj_list = []
    

    def init_line(self, color):
        """ return line object with color from self.plot_color list """
        return self.view.plot(pen=pg.mkPen(color=color, width=1.5))

    def init_dot(self, color):
        """ return dots object with color from self.plot_color list """
        return self.view.plot(pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=color)
    
    def init_line_graphs(self, data_classes=1):
        """ Initialize up to 10 objects corresponding to data classes for plotting line.
        data_classes - amount of different data represented on one graph."""
        
        for i in range(data_classes):
            self.line_obj_list.append(self.init_line(self.plot_color[i]))
    
    def init_dots_graphs(self, data_classes=1):
        """ Initialize up to 10 objects corresponding to data classes for plotting dots.
        data_classes - amount of different data represented on one graph."""

        for i in range(data_classes):
            self.dots_obj_list.append(self.init_dot(self.plot_color[i]))


    def get_dots_list(self):
        """ Get list of all dots. Position corresponds to data class. """
        return self.dots_obj_list
    
    def get_line(self):
        """ Get list of all lines. Position corresponds to data class """
        return self.line_obj_list


    def plot_dots_single_class(self, sequence_num=0, x=[0], y=[0]):
        """ Plot dot object from the list at position sequence_num. """
        self.dots_obj_list[sequence_num].setData(x, y)
    
    def plot_line(self, sequence_num=0, x=[0], y=[0]):
        """ Plot line object from the list at position sequence_num. """
        print(len(self.line_obj_list))
        self.line_obj_list[sequence_num].setData(x, y)
    

    def clear_plot(self):
        """ Erase all from the plot, also deleting all displaying objects. After re-initialize needed. """
        self.view.clear()


if __name__ == "__main__":
    app = Qt.QApplication([])
    w = Graph()
    w.show()
    app.exec()
    pyqtgraph.examples.run()
