import os

import matplotlib.pyplot as plt
import numpy
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox
from keras.models import load_model

from main_GUI import Ui_MainWindow


class Main:
    def __init__(self):
        self.model = None
        self.Class = None
        self.image = None
        self.result = None
        self.percent = None

        self.main_window = QtWidgets.QMainWindow()
        self.main_obj = Ui_MainWindow()
        self.main_obj.setupUi(self.main_window)

        self.main_obj.btn_import_image.clicked.connect(self.load_image)
        self.main_obj.btn_predict_class.clicked.connect(self.show_prediction)
        self.main_obj.btn_show_graph.clicked.connect(self.show_graph)

        self.load_model()

    def load_model(self):
        if os.path.exists('Trained_Model_Cat&Dog.h5'):
            self.model = load_model('Trained_Model_Cat&Dog.h5')
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No model found")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

    def show_graph(self):

        x1 = [0]
        x2 = [1]
        if self.result == 0:
            y1 = self.percent[0]
            y2 = self.percent[1]
        else:
            y1 = self.percent[1]
            y2 = self.percent[0]

        plt.bar(x1, y1, label="cat")
        plt.bar(x2, y2, label="dog")

        plt.ylim((0, 100))
        plt.legend()
        if os.path.exists("graph.jpg"):
            os.remove("graph.jpg")
        plt.savefig("graph.jpg")
        self.main_obj.label_6.setPixmap(QtGui.QPixmap("graph.jpg"))
        plt.clf()

    def show_prediction(self):

        if self.image == None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No image sample provided")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        image = self.image.resize((128, 128))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        image = image / 255
        result = self.model.predict_classes([image])
        pred = result[0]

        self.percent = self.model.predict([image])
        self.percent = self.percent.tolist()[0]
        self.percent = [int(i * 100) for i in self.percent]

        if pred == 0:
            self.main_obj.Label_show_presiction.setText(
                "Dog:" + str(self.percent[0]) + "%")
        else:
            self.main_obj.Label_show_presiction.setText(
                "Cat:" + str(self.percent[1]) + "%")

    def load_image(self):
        # open the dialogue box to select the file
        options = QtWidgets.QFileDialog.Options()

        # open the Dialogue box to get the images paths
        image = QtWidgets.QFileDialog.getOpenFileName(caption="Select the image of the animal", directory="",
                                                      filter="Image Files (*.jpg);;Image Files (*.png)",
                                                      options=options)
        self.main_obj.label_4.setPixmap(QtGui.QPixmap(image[0]))
        self.image = Image.open(image[0])


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    obj = Main()
    obj.main_window.show()
    sys.exit((app.exec_()))
