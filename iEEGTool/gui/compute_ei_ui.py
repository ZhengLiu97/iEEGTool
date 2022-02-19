# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compute_ei.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(420, 453)
        MainWindow.setMinimumSize(QtCore.QSize(420, 453))
        MainWindow.setMaximumSize(QtCore.QSize(420, 453))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.groupBox.setStyleSheet("")
        self.groupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(15)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self._win_lb = QtWidgets.QLabel(self.groupBox)
        self._win_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._win_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._win_lb.setFont(font)
        self._win_lb.setObjectName("_win_lb")
        self.verticalLayout_3.addWidget(self._win_lb)
        self._lfreq_lb = QtWidgets.QLabel(self.groupBox)
        self._lfreq_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._lfreq_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._lfreq_lb.setFont(font)
        self._lfreq_lb.setObjectName("_lfreq_lb")
        self.verticalLayout_3.addWidget(self._lfreq_lb)
        self._hfreq_lb = QtWidgets.QLabel(self.groupBox)
        self._hfreq_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._hfreq_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._hfreq_lb.setFont(font)
        self._hfreq_lb.setObjectName("_hfreq_lb")
        self.verticalLayout_3.addWidget(self._hfreq_lb)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName("verticalLayout")
        self._win_le = QtWidgets.QLineEdit(self.groupBox)
        self._win_le.setMinimumSize(QtCore.QSize(80, 25))
        self._win_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._win_le.setFont(font)
        self._win_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._win_le.setAlignment(QtCore.Qt.AlignCenter)
        self._win_le.setObjectName("_win_le")
        self.verticalLayout.addWidget(self._win_le)
        self._lfreq_low_le = QtWidgets.QLineEdit(self.groupBox)
        self._lfreq_low_le.setMinimumSize(QtCore.QSize(80, 25))
        self._lfreq_low_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._lfreq_low_le.setFont(font)
        self._lfreq_low_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._lfreq_low_le.setAlignment(QtCore.Qt.AlignCenter)
        self._lfreq_low_le.setObjectName("_lfreq_low_le")
        self.verticalLayout.addWidget(self._lfreq_low_le)
        self._hfreq_low_le = QtWidgets.QLineEdit(self.groupBox)
        self._hfreq_low_le.setMinimumSize(QtCore.QSize(80, 25))
        self._hfreq_low_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._hfreq_low_le.setFont(font)
        self._hfreq_low_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._hfreq_low_le.setAlignment(QtCore.Qt.AlignCenter)
        self._hfreq_low_le.setObjectName("_hfreq_low_le")
        self.verticalLayout.addWidget(self._hfreq_low_le)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(15)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self._step_lb = QtWidgets.QLabel(self.groupBox)
        self._step_lb.setMinimumSize(QtCore.QSize(52, 25))
        self._step_lb.setMaximumSize(QtCore.QSize(52, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._step_lb.setFont(font)
        self._step_lb.setAlignment(QtCore.Qt.AlignCenter)
        self._step_lb.setObjectName("_step_lb")
        self.verticalLayout_4.addWidget(self._step_lb)
        self._dash1_lb = QtWidgets.QLabel(self.groupBox)
        self._dash1_lb.setMinimumSize(QtCore.QSize(52, 25))
        self._dash1_lb.setMaximumSize(QtCore.QSize(52, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._dash1_lb.setFont(font)
        self._dash1_lb.setAlignment(QtCore.Qt.AlignCenter)
        self._dash1_lb.setObjectName("_dash1_lb")
        self.verticalLayout_4.addWidget(self._dash1_lb)
        self._dash2_lb = QtWidgets.QLabel(self.groupBox)
        self._dash2_lb.setMinimumSize(QtCore.QSize(52, 25))
        self._dash2_lb.setMaximumSize(QtCore.QSize(52, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._dash2_lb.setFont(font)
        self._dash2_lb.setAlignment(QtCore.Qt.AlignCenter)
        self._dash2_lb.setObjectName("_dash2_lb")
        self.verticalLayout_4.addWidget(self._dash2_lb)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(15)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self._step_le = QtWidgets.QLineEdit(self.groupBox)
        self._step_le.setMinimumSize(QtCore.QSize(80, 25))
        self._step_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._step_le.setFont(font)
        self._step_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._step_le.setAlignment(QtCore.Qt.AlignCenter)
        self._step_le.setObjectName("_step_le")
        self.verticalLayout_2.addWidget(self._step_le)
        self._lfreq_high_le = QtWidgets.QLineEdit(self.groupBox)
        self._lfreq_high_le.setMinimumSize(QtCore.QSize(80, 25))
        self._lfreq_high_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._lfreq_high_le.setFont(font)
        self._lfreq_high_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._lfreq_high_le.setAlignment(QtCore.Qt.AlignCenter)
        self._lfreq_high_le.setObjectName("_lfreq_high_le")
        self.verticalLayout_2.addWidget(self._lfreq_high_le)
        self._hfreq_high_le = QtWidgets.QLineEdit(self.groupBox)
        self._hfreq_high_le.setMinimumSize(QtCore.QSize(80, 25))
        self._hfreq_high_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._hfreq_high_le.setFont(font)
        self._hfreq_high_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._hfreq_high_le.setAlignment(QtCore.Qt.AlignCenter)
        self._hfreq_high_le.setObjectName("_hfreq_high_le")
        self.verticalLayout_2.addWidget(self._hfreq_high_le)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_7.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAutoFillBackground(True)
        self.groupBox_2.setStyleSheet("")
        self.groupBox_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_5.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self._bias_lb = QtWidgets.QLabel(self.groupBox_2)
        self._bias_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._bias_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._bias_lb.setFont(font)
        self._bias_lb.setObjectName("_bias_lb")
        self.horizontalLayout_3.addWidget(self._bias_lb)
        self._bias_le = QtWidgets.QLineEdit(self.groupBox_2)
        self._bias_le.setMinimumSize(QtCore.QSize(80, 25))
        self._bias_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._bias_le.setFont(font)
        self._bias_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._bias_le.setAlignment(QtCore.Qt.AlignCenter)
        self._bias_le.setObjectName("_bias_le")
        self.horizontalLayout_3.addWidget(self._bias_le)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self._threshold_lb = QtWidgets.QLabel(self.groupBox_2)
        self._threshold_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._threshold_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._threshold_lb.setFont(font)
        self._threshold_lb.setObjectName("_threshold_lb")
        self.horizontalLayout_4.addWidget(self._threshold_lb)
        self._threshold_le = QtWidgets.QLineEdit(self.groupBox_2)
        self._threshold_le.setMinimumSize(QtCore.QSize(80, 25))
        self._threshold_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._threshold_le.setFont(font)
        self._threshold_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._threshold_le.setAlignment(QtCore.Qt.AlignCenter)
        self._threshold_le.setObjectName("_threshold_le")
        self.horizontalLayout_4.addWidget(self._threshold_le)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem4)
        self._decay_lb = QtWidgets.QLabel(self.groupBox_2)
        self._decay_lb.setMinimumSize(QtCore.QSize(50, 25))
        self._decay_lb.setMaximumSize(QtCore.QSize(50, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._decay_lb.setFont(font)
        self._decay_lb.setObjectName("_decay_lb")
        self.horizontalLayout_4.addWidget(self._decay_lb)
        self._decay_le = QtWidgets.QLineEdit(self.groupBox_2)
        self._decay_le.setMinimumSize(QtCore.QSize(80, 25))
        self._decay_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._decay_le.setFont(font)
        self._decay_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._decay_le.setAlignment(QtCore.Qt.AlignCenter)
        self._decay_le.setObjectName("_decay_le")
        self.horizontalLayout_4.addWidget(self._decay_le)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.verticalLayout_7.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setAutoFillBackground(True)
        self.groupBox_3.setStyleSheet("")
        self.groupBox_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_6.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self._chan_lb = QtWidgets.QLabel(self.groupBox_3)
        self._chan_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._chan_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._chan_lb.setFont(font)
        self._chan_lb.setObjectName("_chan_lb")
        self.horizontalLayout_2.addWidget(self._chan_lb)
        self._select_chan_btn = QtWidgets.QToolButton(self.groupBox_3)
        self._select_chan_btn.setMinimumSize(QtCore.QSize(80, 25))
        self._select_chan_btn.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._select_chan_btn.setFont(font)
        self._select_chan_btn.setObjectName("_select_chan_btn")
        self.horizontalLayout_2.addWidget(self._select_chan_btn)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self._duration_lb = QtWidgets.QLabel(self.groupBox_3)
        self._duration_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._duration_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._duration_lb.setFont(font)
        self._duration_lb.setObjectName("_duration_lb")
        self.horizontalLayout_6.addWidget(self._duration_lb)
        self._duration_le = QtWidgets.QLineEdit(self.groupBox_3)
        self._duration_le.setMinimumSize(QtCore.QSize(80, 25))
        self._duration_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._duration_le.setFont(font)
        self._duration_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._duration_le.setAlignment(QtCore.Qt.AlignCenter)
        self._duration_le.setObjectName("_duration_le")
        self.horizontalLayout_6.addWidget(self._duration_le)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self._compute_btn = QtWidgets.QPushButton(self.groupBox_3)
        self._compute_btn.setMinimumSize(QtCore.QSize(110, 25))
        self._compute_btn.setMaximumSize(QtCore.QSize(110, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._compute_btn.setFont(font)
        self._compute_btn.setObjectName("_compute_btn")
        self.horizontalLayout_6.addWidget(self._compute_btn)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self._ez_threshold_lb = QtWidgets.QLabel(self.groupBox_3)
        self._ez_threshold_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._ez_threshold_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._ez_threshold_lb.setFont(font)
        self._ez_threshold_lb.setObjectName("_ez_threshold_lb")
        self.horizontalLayout_7.addWidget(self._ez_threshold_lb)
        self._ez_threshold_le = QtWidgets.QLineEdit(self.groupBox_3)
        self._ez_threshold_le.setMinimumSize(QtCore.QSize(80, 25))
        self._ez_threshold_le.setMaximumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._ez_threshold_le.setFont(font)
        self._ez_threshold_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._ez_threshold_le.setAlignment(QtCore.Qt.AlignCenter)
        self._ez_threshold_le.setObjectName("_ez_threshold_le")
        self.horizontalLayout_7.addWidget(self._ez_threshold_le)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem7)
        self._display_table_btn = QtWidgets.QPushButton(self.groupBox_3)
        self._display_table_btn.setMinimumSize(QtCore.QSize(110, 25))
        self._display_table_btn.setMaximumSize(QtCore.QSize(110, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._display_table_btn.setFont(font)
        self._display_table_btn.setObjectName("_display_table_btn")
        self.horizontalLayout_7.addWidget(self._display_table_btn)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.verticalLayout_7.addWidget(self.groupBox_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self.toolBar.setFont(font)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self._viz_ieeg_action = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../icon/square-wave.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._viz_ieeg_action.setIcon(icon)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self._viz_ieeg_action.setFont(font)
        self._viz_ieeg_action.setObjectName("_viz_ieeg_action")
        self._save_excel_action = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../icon/save.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._save_excel_action.setIcon(icon1)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self._save_excel_action.setFont(font)
        self._save_excel_action.setObjectName("_save_excel_action")
        self._import_ei_action = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../../icon/folder.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._import_ei_action.setIcon(icon2)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self._import_ei_action.setFont(font)
        self._import_ei_action.setObjectName("_import_ei_action")
        self._help_action = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../../icon/help.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._help_action.setIcon(icon3)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self._help_action.setFont(font)
        self._help_action.setObjectName("_help_action")
        self._bar_chart_action = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../../icon/bar-chart.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._bar_chart_action.setIcon(icon4)
        self._bar_chart_action.setObjectName("_bar_chart_action")
        self._3d_vis_action = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../../icon/brain.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._3d_vis_action.setIcon(icon5)
        self._3d_vis_action.setObjectName("_3d_vis_action")
        self.toolBar.addAction(self._import_ei_action)
        self.toolBar.addAction(self._save_excel_action)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self._viz_ieeg_action)
        self.toolBar.addAction(self._bar_chart_action)
        self.toolBar.addAction(self._3d_vis_action)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self._help_action)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Energy Ratio Parameters"))
        self._win_lb.setText(_translate("MainWindow", "Window (s)"))
        self._lfreq_lb.setText(_translate("MainWindow", "Low Freq (Hz)"))
        self._hfreq_lb.setText(_translate("MainWindow", "High Freq (Hz)"))
        self._win_le.setText(_translate("MainWindow", "1"))
        self._lfreq_low_le.setText(_translate("MainWindow", "4"))
        self._hfreq_low_le.setText(_translate("MainWindow", "12"))
        self._step_lb.setText(_translate("MainWindow", "Step (s)"))
        self._dash1_lb.setText(_translate("MainWindow", "—"))
        self._dash2_lb.setText(_translate("MainWindow", "—"))
        self._step_le.setText(_translate("MainWindow", "0.25"))
        self._lfreq_high_le.setText(_translate("MainWindow", "12"))
        self._hfreq_high_le.setText(_translate("MainWindow", "127"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Detection Parameters"))
        self._bias_lb.setText(_translate("MainWindow", "Bias"))
        self._bias_le.setText(_translate("MainWindow", "0.1"))
        self._threshold_lb.setText(_translate("MainWindow", "Threshold"))
        self._threshold_le.setText(_translate("MainWindow", "1"))
        self._decay_lb.setText(_translate("MainWindow", "Decay"))
        self._decay_le.setText(_translate("MainWindow", "1"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Epileptogenicity Index"))
        self._chan_lb.setText(_translate("MainWindow", "Channels"))
        self._select_chan_btn.setText(_translate("MainWindow", "..."))
        self._duration_lb.setText(_translate("MainWindow", "Duration (s)"))
        self._duration_le.setText(_translate("MainWindow", "5"))
        self._compute_btn.setText(_translate("MainWindow", "Compute"))
        self._compute_btn.setShortcut(_translate("MainWindow", "Return"))
        self._ez_threshold_lb.setText(_translate("MainWindow", "EZ threshold"))
        self._ez_threshold_le.setText(_translate("MainWindow", "0.2"))
        self._display_table_btn.setText(_translate("MainWindow", "Display"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self._viz_ieeg_action.setText(_translate("MainWindow", "Vis in iEEG"))
        self._save_excel_action.setText(_translate("MainWindow", "Save EI"))
        self._import_ei_action.setText(_translate("MainWindow", "Import EI"))
        self._help_action.setText(_translate("MainWindow", "Help"))
        self._bar_chart_action.setText(_translate("MainWindow", "BarChart"))
        self._3d_vis_action.setText(_translate("MainWindow", "3D visualization"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
