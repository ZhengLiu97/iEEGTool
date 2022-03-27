# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rois_viz.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1301, 900)
        MainWindow.setMinimumSize(QtCore.QSize(1300, 900))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self._brain_gp = QtWidgets.QGroupBox(self.centralwidget)
        self._brain_gp.setMinimumSize(QtCore.QSize(500, 0))
        self._brain_gp.setMaximumSize(QtCore.QSize(500, 16777215))
        self._brain_gp.setCheckable(True)
        self._brain_gp.setObjectName("_brain_gp")
        self.verticalLayout = QtWidgets.QVBoxLayout(self._brain_gp)
        self.verticalLayout.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self._transparency_lb = QtWidgets.QLabel(self._brain_gp)
        self._transparency_lb.setMinimumSize(QtCore.QSize(75, 25))
        self._transparency_lb.setMaximumSize(QtCore.QSize(16777215, 25))
        self._transparency_lb.setObjectName("_transparency_lb")
        self.horizontalLayout.addWidget(self._transparency_lb)
        self._transparency_slider = QtWidgets.QSlider(self._brain_gp)
        self._transparency_slider.setMinimumSize(QtCore.QSize(250, 25))
        self._transparency_slider.setMaximumSize(QtCore.QSize(250, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self._transparency_slider.setFont(font)
        self._transparency_slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._transparency_slider.setAutoFillBackground(False)
        self._transparency_slider.setMaximum(100)
        self._transparency_slider.setSingleStep(10)
        self._transparency_slider.setPageStep(10)
        self._transparency_slider.setProperty("value", 10)
        self._transparency_slider.setSliderPosition(10)
        self._transparency_slider.setOrientation(QtCore.Qt.Horizontal)
        self._transparency_slider.setInvertedAppearance(False)
        self._transparency_slider.setInvertedControls(False)
        self._transparency_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self._transparency_slider.setTickInterval(10)
        self._transparency_slider.setObjectName("_transparency_slider")
        self.horizontalLayout.addWidget(self._transparency_slider)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self._hemi_lb = QtWidgets.QLabel(self._brain_gp)
        self._hemi_lb.setMinimumSize(QtCore.QSize(113, 25))
        self._hemi_lb.setMaximumSize(QtCore.QSize(1999999, 25))
        self._hemi_lb.setObjectName("_hemi_lb")
        self.horizontalLayout_2.addWidget(self._hemi_lb)
        self._hemi_cbx = QtWidgets.QComboBox(self._brain_gp)
        self._hemi_cbx.setMinimumSize(QtCore.QSize(250, 25))
        self._hemi_cbx.setMaximumSize(QtCore.QSize(1000, 25))
        self._hemi_cbx.setObjectName("_hemi_cbx")
        self._hemi_cbx.addItem("")
        self._hemi_cbx.addItem("")
        self._hemi_cbx.addItem("")
        self.horizontalLayout_2.addWidget(self._hemi_cbx)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3.addWidget(self._brain_gp)
        self._rois_gp = QtWidgets.QGroupBox(self.centralwidget)
        self._rois_gp.setMinimumSize(QtCore.QSize(500, 700))
        self._rois_gp.setMaximumSize(QtCore.QSize(500, 16777215))
        self._rois_gp.setCheckable(True)
        self._rois_gp.setObjectName("_rois_gp")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self._rois_gp)
        self.verticalLayout_2.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout_2.setSpacing(7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self._chs_cbx = QtWidgets.QCheckBox(self._rois_gp)
        self._chs_cbx.setObjectName("_chs_cbx")
        self.horizontalLayout_6.addWidget(self._chs_cbx)
        self._chs_name_cbx = QtWidgets.QCheckBox(self._rois_gp)
        self._chs_name_cbx.setEnabled(False)
        self._chs_name_cbx.setObjectName("_chs_name_cbx")
        self.horizontalLayout_6.addWidget(self._chs_name_cbx)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self._info_table = QtWidgets.QTableWidget(self._rois_gp)
        self._info_table.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(9)
        self._info_table.setFont(font)
        self._info_table.setWordWrap(False)
        self._info_table.setObjectName("_info_table")
        self._info_table.setColumnCount(0)
        self._info_table.setRowCount(0)
        self.verticalLayout_2.addWidget(self._info_table)
        self.verticalLayout_3.addWidget(self._rois_gp)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self._plotter = Brain(self.centralwidget)
        self._plotter.setMinimumSize(QtCore.QSize(770, 800))
        self._plotter.setObjectName("_plotter")
        self.horizontalLayout_3.addWidget(self._plotter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1301, 26))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        self._View_menu = QtWidgets.QMenu(self.menuEdit)
        self._View_menu.setObjectName("_View_menu")
        self._Figure_menu = QtWidgets.QMenu(self.menuEdit)
        self._Figure_menu.setObjectName("_Figure_menu")
        MainWindow.setMenuBar(self.menuBar)
        self.actionScreenshot = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../icon/screenshot.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionScreenshot.setIcon(icon)
        self.actionScreenshot.setObjectName("actionScreenshot")
        self._screenshot_action = QtWidgets.QAction(MainWindow)
        self._screenshot_action.setObjectName("_screenshot_action")
        self.actionBrain_Color = QtWidgets.QAction(MainWindow)
        self.actionBrain_Color.setObjectName("actionBrain_Color")
        self.actionBackground_Color = QtWidgets.QAction(MainWindow)
        self.actionBackground_Color.setObjectName("actionBackground_Color")
        self._background_color_action = QtWidgets.QAction(MainWindow)
        self._background_color_action.setObjectName("_background_color_action")
        self._brain_color_action = QtWidgets.QAction(MainWindow)
        self._brain_color_action.setObjectName("_brain_color_action")
        self._front_action = QtWidgets.QAction(MainWindow)
        self._front_action.setObjectName("_front_action")
        self._back_action = QtWidgets.QAction(MainWindow)
        self._back_action.setObjectName("_back_action")
        self._left_action = QtWidgets.QAction(MainWindow)
        self._left_action.setObjectName("_left_action")
        self._right_action = QtWidgets.QAction(MainWindow)
        self._right_action.setObjectName("_right_action")
        self._top_action = QtWidgets.QAction(MainWindow)
        self._top_action.setObjectName("_top_action")
        self._bottom_action = QtWidgets.QAction(MainWindow)
        self._bottom_action.setObjectName("_bottom_action")
        self.menuFile.addAction(self._screenshot_action)
        self._View_menu.addAction(self._front_action)
        self._View_menu.addAction(self._back_action)
        self._View_menu.addAction(self._left_action)
        self._View_menu.addAction(self._right_action)
        self._View_menu.addAction(self._top_action)
        self._View_menu.addAction(self._bottom_action)
        self._Figure_menu.addAction(self._background_color_action)
        self._Figure_menu.addAction(self._brain_color_action)
        self.menuEdit.addAction(self._Figure_menu.menuAction())
        self.menuEdit.addAction(self._View_menu.menuAction())
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self._brain_gp.setTitle(_translate("MainWindow", "Brain"))
        self._transparency_lb.setText(_translate("MainWindow", "Transparency"))
        self._hemi_lb.setText(_translate("MainWindow", "Hemisphere"))
        self._hemi_cbx.setItemText(0, _translate("MainWindow", "Both"))
        self._hemi_cbx.setItemText(1, _translate("MainWindow", "Left"))
        self._hemi_cbx.setItemText(2, _translate("MainWindow", "Right"))
        self._rois_gp.setTitle(_translate("MainWindow", "Regions of Interest"))
        self._chs_cbx.setText(_translate("MainWindow", "Elelctrodes of Interest"))
        self._chs_name_cbx.setText(_translate("MainWindow", "Show Electrodes\' name"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self._View_menu.setTitle(_translate("MainWindow", "View"))
        self._Figure_menu.setTitle(_translate("MainWindow", "Figure"))
        self.actionScreenshot.setText(_translate("MainWindow", "Screenshot"))
        self._screenshot_action.setText(_translate("MainWindow", "Screenshot"))
        self.actionBrain_Color.setText(_translate("MainWindow", "Brain Color"))
        self.actionBackground_Color.setText(_translate("MainWindow", "Background Color"))
        self._background_color_action.setText(_translate("MainWindow", "Background color"))
        self._brain_color_action.setText(_translate("MainWindow", "Brain color"))
        self._front_action.setText(_translate("MainWindow", "Front"))
        self._back_action.setText(_translate("MainWindow", "Back"))
        self._left_action.setText(_translate("MainWindow", "Left"))
        self._right_action.setText(_translate("MainWindow", "Right"))
        self._top_action.setText(_translate("MainWindow", "Top"))
        self._bottom_action.setText(_translate("MainWindow", "Bottom"))
from viz.brain import Brain


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
