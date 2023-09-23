from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from mygui import Ui_ReidSystem
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from visualization_one_pic_gui import predict
from threading import Thread
from PyQt5.QtCore import QObject, pyqtSignal

class MySignals(QObject):
    pic_print = pyqtSignal(QLabel, QPixmap)  # 打印图片的信号
    text_print = pyqtSignal(QLabel, str)  # 打印字符的信号
    browse_print = pyqtSignal(QLabel, str)

class Camshow(QMainWindow, Ui_ReidSystem):
    def __init__(self, parent=None):
        super(Camshow, self).__init__(parent)
        self.setupUi(self)
        self.getphoto.clicked.connect(self.loadImage)
        self.dopredict.clicked.connect(self.predict_thread)
        self.setFixedSize(self.width(), self.height())
        self.ms = MySignals()  # 实例化一个signal
        self.ms.pic_print.connect(self.PrintPicToGui)
        self.ms.text_print.connect(self.PrintTextToGui)
        self.ms.browse_print.connect(self.PrintCommandToGui)
        pic_opendir = 'opendir.png'
        self.getphoto.setStyleSheet("QPushButton{\n"
                                    "background-image: url(\"%s\");\n"
                                    "background-position:left;\n"
                                    "background-repeat:no-repeat;\n"
                                    "}" % pic_opendir)
        pic_dopredict = 'search.png'
        self.dopredict.setStyleSheet("QPushButton{\n"
                                    "background-image: url(\"%s\");\n"
                                    "background-position:left;\n"
                                    "background-repeat:no-repeat;\n"
                                    "}" % pic_dopredict)

    def PrintPicToGui(self, fb, pic):
        fb.setPixmap(pic)

    def PrintTextToGui(self, fb, text):
        fb.setText(text)

    def PrintCommandToGui(self, fb, text):
        fb.append(str(text))
        fb.ensureCursorVisible()


    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileNames(self, '选择图片', r'D:\pythoncode\NWPU-ReID_old\ori_data',
                                                     '图像文件(*.jpg *.jpeg *.png)')
        if self.fname:
            pic = QPixmap(*self.fname).scaled(self.query.width(), self.query.height())
            img_name = self.fname[0]
            self.queryID.setText('ID:' + str(int(img_name[-43:-39])))  # 显示query ID号
            self.query.setPixmap(pic)
            self.command.clear()
            self.ms.browse_print.emit(self.command, '图片路径： ' + img_name)

    def get_info(self, rankn,picpath):
        time = picpath[-33:-14].split('_')
        cam = picpath[-13]
        id = picpath[-38:-34]
        return 'Rank-{}行人编号{},{}号摄像机,拍摄时间: {}年{}月{}日{}时{}分{}秒'.format((str(rankn)+',').ljust(3,' '), id, cam,
                                                                             time[0], time[1], time[2],
                                                                             time[3], time[4], time[5])

    def predict_thread(self):
        def run():
            if self.mode.currentText() == '可见光搜索红外':
                mode = 'v2t'
                self.ms.browse_print.emit(self.command,
                                          '---------------------------------------------------------------')
                self.ms.browse_print.emit(self.command, '搜索模式: \'可见光搜索红外\'.')

            if self.mode.currentText() == '红外搜索可见光':
                mode = 't2v'
                self.ms.browse_print.emit(self.command,
                                          '---------------------------------------------------------------')
                self.ms.browse_print.emit(self.command, '搜索模式: \'红外搜索可见光\'')
            self.ms.browse_print.emit(self.command,
                                      '---------------------------------------------------------------')
            self.ms.browse_print.emit(self.command, '开始处理，请等待......')
            target_gallery_path, labeltext, similartiytext = predict(*self.fname,
                                                                     resume='nwpu_agw_p4_n8_lr_0.1_seed_0_best.t',
                                                                     mode=mode)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[0]).scaled(self.r1photo.width(), self.r1photo.height())
            self.ms.browse_print.emit(self.command,
                                      '---------------------------------------------------------------')
            self.ms.pic_print.emit(self.r1photo, pic)
            self.ms.text_print.emit(self.r1text,'ID:'+labeltext[0])
            self.ms.text_print.emit(self.r1text_2, similartiytext[0])
            information = self.get_info(1, target_gallery_path[0])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[1]).scaled(self.r2photo.width(), self.r2photo.height())
            self.ms.pic_print.emit(self.r2photo, pic)
            self.ms.text_print.emit(self.r2text,'ID:'+labeltext[1])
            self.ms.text_print.emit(self.r2text_2, similartiytext[1])
            information = self.get_info(2, target_gallery_path[1])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[2]).scaled(self.r3photo.width(), self.r3photo.height())
            self.ms.pic_print.emit(self.r3photo, pic)
            self.ms.text_print.emit(self.r3text,'ID:'+labeltext[2])
            self.ms.text_print.emit(self.r3text_2, similartiytext[2])
            information = self.get_info(3, target_gallery_path[2])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[3]).scaled(self.r4photo.width(), self.r4photo.height())
            self.ms.pic_print.emit(self.r4photo, pic)
            self.ms.text_print.emit(self.r4text,'ID:'+labeltext[3])
            self.ms.text_print.emit(self.r4text_2, similartiytext[3])
            information = self.get_info(4, target_gallery_path[3])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[4]).scaled(self.r5photo.width(), self.r5photo.height())
            self.ms.pic_print.emit(self.r5photo, pic)
            self.ms.text_print.emit(self.r5text,'ID:'+labeltext[4])
            self.ms.text_print.emit(self.r5text_2, similartiytext[4])
            information = self.get_info(5, target_gallery_path[4])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[5]).scaled(self.r6photo.width(), self.r6photo.height())
            self.ms.pic_print.emit(self.r6photo, pic)
            self.ms.text_print.emit(self.r6text,'ID:'+labeltext[5])
            self.ms.text_print.emit(self.r6text_2, similartiytext[5])
            information = self.get_info(6, target_gallery_path[5])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[6]).scaled(self.r7photo.width(), self.r7photo.height())
            self.ms.pic_print.emit(self.r7photo, pic)
            self.ms.text_print.emit(self.r7text,'ID:'+labeltext[6])
            self.ms.text_print.emit(self.r7text_2, similartiytext[6])
            information = self.get_info(7, target_gallery_path[6])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[7]).scaled(self.r8photo.width(), self.r8photo.height())
            self.ms.pic_print.emit(self.r8photo, pic)
            self.ms.text_print.emit(self.r8text,'ID:'+labeltext[7])
            self.ms.text_print.emit(self.r8text_2, similartiytext[7])
            information = self.get_info(8, target_gallery_path[7])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[8]).scaled(self.r9photo.width(), self.r9photo.height())
            self.ms.pic_print.emit(self.r9photo, pic)
            self.ms.text_print.emit(self.r9text,'ID:'+labeltext[8])
            self.ms.text_print.emit(self.r9text_2, similartiytext[8])
            information = self.get_info(9, target_gallery_path[8])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            pic = QPixmap(target_gallery_path[9]).scaled(self.r10photo.width(), self.r10photo.height())
            self.ms.pic_print.emit(self.r10photo, pic)
            self.ms.text_print.emit(self.r10text,'ID:'+labeltext[9])
            self.ms.text_print.emit(self.r10text_2, similartiytext[9])
            information = self.get_info(10, target_gallery_path[9])
            self.ms.browse_print.emit(self.command, information)
            # --------------------------------------------------------------------------------------------
            self.ms.browse_print.emit(self.command,
                                      '---------------------------------------------------------------')
            self.ms.browse_print.emit(self.command, '完成! \'运行结果\' 展示了相似度排名前十的图片.')
        t = Thread(target=run)
        t.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('exeicon.png'))
    ui = Camshow()
    ui.show()
    sys.exit(app.exec_())
