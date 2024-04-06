# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mw.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1109, 825)
        MainWindow.setMinimumSize(QtCore.QSize(1040, 825))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 760, 760))
        self.graphicsView.setObjectName("graphicsView")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(830, 0, 271, 760))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_graficar = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_graficar.setGeometry(QtCore.QRect(10, 640, 221, 51))
        self.pushButton_graficar.setObjectName("pushButton_graficar")
        self.pushButton_reset = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_reset.setGeometry(QtCore.QRect(10, 699, 221, 51))
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 30, 231, 471))
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_bias = QtWidgets.QLabel(self.groupBox_2)
        self.label_bias.setObjectName("label_bias")
        self.verticalLayout.addWidget(self.label_bias)
        self.lineEdit_bias = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_bias.setEnabled(True)
        self.lineEdit_bias.setReadOnly(True)
        self.lineEdit_bias.setObjectName("lineEdit_bias")
        self.verticalLayout.addWidget(self.lineEdit_bias)
        self.label_w1 = QtWidgets.QLabel(self.groupBox_2)
        self.label_w1.setObjectName("label_w1")
        self.verticalLayout.addWidget(self.label_w1)
        self.lineEdit_w1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_w1.setReadOnly(True)
        self.lineEdit_w1.setObjectName("lineEdit_w1")
        self.verticalLayout.addWidget(self.lineEdit_w1)
        self.label_w2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_w2.setObjectName("label_w2")
        self.verticalLayout.addWidget(self.label_w2)
        self.lineEdit_w2 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_w2.setReadOnly(True)
        self.lineEdit_w2.setObjectName("lineEdit_w2")
        self.verticalLayout.addWidget(self.lineEdit_w2)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.lineEdit_factor = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_factor.setObjectName("lineEdit_factor")
        self.verticalLayout.addWidget(self.lineEdit_factor)
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.lineEdit_limite = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_limite.setObjectName("lineEdit_limite")
        self.verticalLayout.addWidget(self.lineEdit_limite)
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.lineEdit_restantes = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_restantes.setReadOnly(True)
        self.lineEdit_restantes.setObjectName("lineEdit_restantes")
        self.verticalLayout.addWidget(self.lineEdit_restantes)
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.lineEdit_error = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_error.setReadOnly(True)
        self.lineEdit_error.setObjectName("lineEdit_error")
        self.verticalLayout.addWidget(self.lineEdit_error)
        self.pushButton_exportar = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_exportar.setGeometry(QtCore.QRect(10, 520, 221, 51))
        self.pushButton_exportar.setObjectName("pushButton_exportar")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 580, 221, 51))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1109, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Graficar"))
        self.pushButton_graficar.setText(_translate("MainWindow", "Graficar"))
        self.pushButton_reset.setText(_translate("MainWindow", "Reset"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Entradas"))
        self.label_bias.setText(_translate("MainWindow", "Bias"))
        self.label_w1.setText(_translate("MainWindow", "W1"))
        self.label_w2.setText(_translate("MainWindow", "W2"))
        self.label_5.setText(_translate("MainWindow", "Factor de aprendizaje"))
        self.label_6.setText(_translate("MainWindow", "Limite de epocas"))
        self.label_7.setText(_translate("MainWindow", "Epocas Restantes"))
        self.label_8.setText(_translate("MainWindow", "Error"))
        self.pushButton_exportar.setText(_translate("MainWindow", "Abrir un archivo(entradas)"))
        self.pushButton.setText(_translate("MainWindow", "Salidas deseadas"))
