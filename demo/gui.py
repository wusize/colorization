import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from demo.Demo import Picture
from demo.Demo import Picturestyle


# ----------------------------------
# 定义Example类
# ----------------------------------

class Example(QWidget):

    def __init__(self):

        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('展示')  # 窗口名称
        self.resize(810, 600)
        # ----------------------------------
        # 加入各个查询按钮
        # ----------------------------------
        self.bt1 = QPushButton('选择图片', self)# 选择图片按钮
        self.bt1.setGeometry(10, 20, 250, 30)
        self.bt2 = QPushButton('添加颜色', self)  # 添加颜色按钮
        self.bt2.setGeometry(280, 20, 250, 30)
        self.bt3 = QPushButton('风格迁移', self)  # 风格迁移按钮
        self.bt3.setGeometry(550, 20, 250, 30)
        # ----------------------------------
        # 连接各个事件与信号槽
        # ----------------------------------
        self.bt1.clicked.connect(self.choiceimage)  # 选择图片信号
        self.bt2.clicked.connect(self.drawcolor)    # 添加颜色信号
        self.bt3.clicked.connect(self.formmove)     # 风格迁移信号
        # ----------------------------------
        # 在label上显示图片
        # ----------------------------------
        self.choicelable = QLabel(self)
        #self.choicelable.resize(200, 100)

        self.colorlabel = QLabel(self)
        self.graylabel = QLabel(self)
        self.anotherlabel = QLabel(self)
        self.formlabel = QLabel(self)

        self.pic = ''
        self.picstyle = ''

        self.show()
        self.center()

    # 询问是否确定退出
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认', '确认退出吗', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 将页面显示在中央
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 选择图片函数
    def choiceimage(self):
        image_file, imgtype= QFileDialog.getOpenFileName(self, 'Open file',
                                                         "/media/wusize/Portable/dataset/Colorization/val/",
                                                         '*.jpg  *.png *.jpeg')
        self.pic = image_file
        print(self.pic)
        imagein = QImage()
        imagein.load(image_file)
        self.choicelable.setGeometry(10, 200, 250, 200)
        self.choicelable.setStyleSheet("border: 2px solid red")
        self.choicelable.setPixmap(QPixmap.fromImage(imagein))
        self.choicelable.setScaledContents(True)

    def drawcolor(self):
        filegray, filecolor = Picture(self.pic)
        print(filegray)
        print(filecolor)
        imagegray = QImage()
        imagegray.load(filegray)
        imagecolor = QImage()
        imagecolor.load(filecolor)
        self.graylabel.setGeometry(280, 100, 250, 200)
        self.graylabel.setStyleSheet("border: 2px solid red")
        self.graylabel.setPixmap(QPixmap.fromImage(imagegray))
        self.graylabel.setScaledContents(True)
        self.colorlabel.setGeometry(280, 320, 250, 200)
        self.colorlabel.setStyleSheet("border: 2px solid red")
        self.colorlabel.setPixmap(QPixmap.fromImage(imagecolor))
        self.colorlabel.setScaledContents(True)

    def formmove(self):
        image_file, imgtype = QFileDialog.getOpenFileName(self, 'Open file',
                                                          "/media/wusize/Portable/dataset/Colorization/val/",
                                                          '*.jpg  *.png *.jpeg')
        self.picstyle = image_file
        if self.picstyle is not '':
            fileform = Picturestyle(self.pic, self.picstyle)
            imagein = QImage()
            imagein.load(image_file)
            imageform = QImage()
            imageform.load(fileform)
            self.anotherlabel.setGeometry(550, 100, 250, 200)
            self.anotherlabel.setStyleSheet("border: 2px solid red")
            self.anotherlabel.setPixmap(QPixmap.fromImage(imagein))
            self.anotherlabel.setScaledContents(True)
            self.formlabel.setGeometry(550, 320, 250, 200)
            self.formlabel.setStyleSheet("border: 2px solid red")
            self.formlabel.setPixmap(QPixmap.fromImage(imageform))
            self.formlabel.setScaledContents(True)
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
