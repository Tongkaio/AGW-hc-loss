from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from mygui import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from visualization_one_pic_gui import predict
from threading import Thread
from PyQt5.QtCore import QObject, pyqtSignal

class Camshow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Camshow, self).__init__(parent)
        self.setupUi(self)
        self.getphoto.clicked.connect(self.loadImage)

    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileNames(self, '选择图片', '.', '图像文件(*.jpg *.jpeg *.png)')
        if self.fname:
            print(self.fname)
            pic = QPixmap(*self.fname).scaled(self.query.width(), self.query.height())
            self.query.setPixmap(pic)
            self.predict_thread()

    def predict_thread(self):
        def run():
            print(*self.fname[-43:-39])
            target_gallery_path, labelwithsimilar = predict(*self.fname, 'nwpu_agw_p4_n8_lr_0.1_seed_0_best.t')
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[0]).scaled(self.r1photo.width(), self.r1photo.height())
            self.r1photo.setPixmap(pic)
            self.r1text.setText(labelwithsimilar[0])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[1]).scaled(self.r2photo.width(), self.r2photo.height())
            self.r2photo.setPixmap(pic)
            self.r2text.setText(labelwithsimilar[1])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[2]).scaled(self.r3photo.width(), self.r3photo.height())
            self.r3photo.setPixmap(pic)
            self.r3text.setText(labelwithsimilar[2])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[3]).scaled(self.r4photo.width(), self.r4photo.height())
            self.r4photo.setPixmap(pic)
            self.r4text.setText(labelwithsimilar[3])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[4]).scaled(self.r5photo.width(), self.r5photo.height())
            self.r5photo.setPixmap(pic)
            self.r5text.setText(labelwithsimilar[4])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[5]).scaled(self.r6photo.width(), self.r6photo.height())
            self.r6photo.setPixmap(pic)
            self.r6text.setText(labelwithsimilar[5])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[6]).scaled(self.r7photo.width(), self.r7photo.height())
            self.r7photo.setPixmap(pic)
            self.r7text.setText(labelwithsimilar[6])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[7]).scaled(self.r8photo.width(), self.r8photo.height())
            self.r8photo.setPixmap(pic)
            self.r8text.setText(labelwithsimilar[7])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[8]).scaled(self.r9photo.width(), self.r9photo.height())
            self.r9photo.setPixmap(pic)
            self.r9text.setText(labelwithsimilar[8])
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[9]).scaled(self.r10photo.width(), self.r10photo.height())
            self.r10photo.setPixmap(pic)
            self.r10text.setText(labelwithsimilar[9])
            # --------------------------------------------------------------------------------------------
            # print(target_gallery_path)
            # print(labelwithsimilar)
        t = Thread(target=run)
        t.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Camshow()
    ui.show()
    sys.exit(app.exec_())
