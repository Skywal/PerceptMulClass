#  Widget class that displays graph.

from PyQt5 import Qt
import numpy as np

import pyqtgraph as pg
import pyqtgraph.examples


class Graph(Qt.QWidget):

    def __init__(self):
        super().__init__()

        layout = Qt.QVBoxLayout(self) # create layout

        self.view = self.init_plot()
        
        layout.addWidget(self.view) # show up plot area
        
        self.init_line()
        self.init_dots_graphs()


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
    
    def init_line(self):
        """ Initialise line-like graph object. """
        self.line_pen = pg.mkPen(color='m', width=1.5)
        self.trand_line = self.view.plot(pen=self.line_pen) 
    
    def init_dots_graphs(self):
        """ Initialise dots-like gaph objects. They are reusable. """
        
        self.first_dot_cloud = self.view.plot(pen=None, symbol='o', symbolPen=None,
                                            symbolSize=5, symbolBrush='b')

        self.second_dot_cloud = self.view.plot(pen=None, symbol='t', symbolPen=None,
                                            symbolSize=5, symbolBrush='r')

        self.third_dot_cloud = self.view.plot(pen=None, symbol='s', symbolPen=None,
                                            symbolSize=5, symbolBrush='g')


    def get_b_dots(self):
        """ Get object for blue dots (1 data class). """
        return self.first_dot_cloud
    
    def get_r_dots(self):
        """ Get object for red dots (2 data class). """
        return self.second_dot_cloud

    def get_g_dots(self):
        """ Get object fot green dots (3 data class). """
        return self.third_dot_cloud

    def get_line(self):
        """ Get object for line. """
        return self.trand_line


    def clear_plot(self):
        """ Erase all from the plot, also deleting all displaying objects. After re-initialize needed. """
        self.view.clear()


    def plot_f_d_class_dots(self,x=[0], y=[0]):
        """ Draw dot like graph with dots coord 'x' and 'y'. 'o'-like dots. First data class. """
        self.first_dot_cloud.setData(x, y)
    
    def plot_s_d_class_dots(self,x=[0], y=[0]):
        """ Draw dot like graph with dots coord 'x' and 'y'. Triangular dots. Second data class. """
        self.second_dot_cloud.setData(x, y)

    def plot_t_d_class_dots(self, x=[0], y=[0]):
        """ Draw dot like graph with dots coord 'x' and 'y'. Square dots. Third data class. """
        self.third_dot_cloud.setData(x, y)


    def plot_line(self, x=[0], y=[0]):
        """Draw line graph with dots coord 'x' and 'y'"""
        self.trand_line.setData(x, y)


if __name__ == "__main__":
    app = Qt.QApplication([])
    w = Graph()
    w.show()
    app.exec()
    pyqtgraph.examples.run()