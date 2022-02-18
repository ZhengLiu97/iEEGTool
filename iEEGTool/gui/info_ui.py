# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'info.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(331, 113)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self._name_lb = QtWidgets.QLabel(self.centralwidget)
        self._name_lb.setMinimumSize(QtCore.QSize(150, 25))
        self._name_lb.setMaximumSize(QtCore.QSize(150, 25))
        self._name_lb.setObjectName("_name_lb")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self._name_lb)
        self._name_le = QtWidgets.QLineEdit(self.centralwidget)
        self._name_le.setMinimumSize(QtCore.QSize(150, 25))
        self._name_le.setMaximumSize(QtCore.QSize(150, 25))
        self._name_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._name_le.setAlignment(QtCore.Qt.AlignCenter)
        self._name_le.setObjectName("_name_le")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self._name_le)
        self._age_lb = QtWidgets.QLabel(self.centralwidget)
        self._age_lb.setMinimumSize(QtCore.QSize(150, 25))
        self._age_lb.setMaximumSize(QtCore.QSize(150, 25))
        self._age_lb.setObjectName("_age_lb")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self._age_lb)
        self._age_le = QtWidgets.QLineEdit(self.centralwidget)
        self._age_le.setMinimumSize(QtCore.QSize(150, 25))
        self._age_le.setMaximumSize(QtCore.QSize(150, 25))
        self._age_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._age_le.setAlignment(QtCore.Qt.AlignCenter)
        self._age_le.setObjectName("_age_le")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self._age_le)
        self._gender_lb = QtWidgets.QLabel(self.centralwidget)
        self._gender_lb.setMinimumSize(QtCore.QSize(150, 25))
        self._gender_lb.setMaximumSize(QtCore.QSize(150, 25))
        self._gender_lb.setObjectName("_gender_lb")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self._gender_lb)
        self._gender_le = QtWidgets.QLineEdit(self.centralwidget)
        self._gender_le.setMinimumSize(QtCore.QSize(150, 25))
        self._gender_le.setMaximumSize(QtCore.QSize(150, 25))
        self._gender_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._gender_le.setAlignment(QtCore.Qt.AlignCenter)
        self._gender_le.setObjectName("_gender_le")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self._gender_le)
        self.verticalLayout.addLayout(self.formLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self._name_lb.setText(_translate("MainWindow", "Subject  Name"))
        self._age_lb.setText(_translate("MainWindow", "Subject  Age"))
        self._gender_lb.setText(_translate("MainWindow", "Subject  Gender"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
