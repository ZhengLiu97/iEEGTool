# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tfr_morlet.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(811, 619)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_8.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout_8.setSpacing(7)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Morlet_methodlb = QtWidgets.QLabel(self.groupBox)
        self.Morlet_methodlb.setMinimumSize(QtCore.QSize(0, 25))
        self.Morlet_methodlb.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.Morlet_methodlb.setFont(font)
        self.Morlet_methodlb.setObjectName("Morlet_methodlb")
        self.horizontalLayout.addWidget(self.Morlet_methodlb)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self._compute_btn = QtWidgets.QPushButton(self.groupBox)
        self._compute_btn.setMinimumSize(QtCore.QSize(100, 25))
        self._compute_btn.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._compute_btn.setFont(font)
        self._compute_btn.setObjectName("_compute_btn")
        self.horizontalLayout.addWidget(self._compute_btn)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.line_5 = QtWidgets.QFrame(self.groupBox)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_8.addWidget(self.line_5)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(35)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(15)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self._chan_lb = QtWidgets.QLabel(self.groupBox)
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
        self.verticalLayout_2.addWidget(self._chan_lb)
        self._freq_band_lb = QtWidgets.QLabel(self.groupBox)
        self._freq_band_lb.setMinimumSize(QtCore.QSize(0, 25))
        self._freq_band_lb.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._freq_band_lb.setFont(font)
        self._freq_band_lb.setObjectName("_freq_band_lb")
        self.verticalLayout_2.addWidget(self._freq_band_lb)
        self._freq_step_lb = QtWidgets.QLabel(self.groupBox)
        self._freq_step_lb.setMinimumSize(QtCore.QSize(0, 25))
        self._freq_step_lb.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._freq_step_lb.setFont(font)
        self._freq_step_lb.setObjectName("_freq_step_lb")
        self.verticalLayout_2.addWidget(self._freq_step_lb)
        self._denom_ncycles_lb = QtWidgets.QLabel(self.groupBox)
        self._denom_ncycles_lb.setMinimumSize(QtCore.QSize(0, 25))
        self._denom_ncycles_lb.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._denom_ncycles_lb.setFont(font)
        self._denom_ncycles_lb.setObjectName("_denom_ncycles_lb")
        self.verticalLayout_2.addWidget(self._denom_ncycles_lb)
        self._njobs_lb = QtWidgets.QLabel(self.groupBox)
        self._njobs_lb.setMinimumSize(QtCore.QSize(0, 25))
        self._njobs_lb.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._njobs_lb.setFont(font)
        self._njobs_lb.setObjectName("_njobs_lb")
        self.verticalLayout_2.addWidget(self._njobs_lb)
        self._log_freq_lb = QtWidgets.QLabel(self.groupBox)
        self._log_freq_lb.setMinimumSize(QtCore.QSize(0, 25))
        self._log_freq_lb.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._log_freq_lb.setFont(font)
        self._log_freq_lb.setObjectName("_log_freq_lb")
        self.verticalLayout_2.addWidget(self._log_freq_lb)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName("verticalLayout")
        self._select_chan_btn = QtWidgets.QToolButton(self.groupBox)
        self._select_chan_btn.setMinimumSize(QtCore.QSize(100, 25))
        self._select_chan_btn.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._select_chan_btn.setFont(font)
        self._select_chan_btn.setObjectName("_select_chan_btn")
        self.verticalLayout.addWidget(self._select_chan_btn)
        self._freq_band_le = QtWidgets.QLineEdit(self.groupBox)
        self._freq_band_le.setMinimumSize(QtCore.QSize(100, 25))
        self._freq_band_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._freq_band_le.setFont(font)
        self._freq_band_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._freq_band_le.setAlignment(QtCore.Qt.AlignCenter)
        self._freq_band_le.setObjectName("_freq_band_le")
        self.verticalLayout.addWidget(self._freq_band_le)
        self._freq_step_le = QtWidgets.QLineEdit(self.groupBox)
        self._freq_step_le.setMinimumSize(QtCore.QSize(100, 25))
        self._freq_step_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._freq_step_le.setFont(font)
        self._freq_step_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._freq_step_le.setAlignment(QtCore.Qt.AlignCenter)
        self._freq_step_le.setObjectName("_freq_step_le")
        self.verticalLayout.addWidget(self._freq_step_le)
        self._denom_ncycles_le = QtWidgets.QLineEdit(self.groupBox)
        self._denom_ncycles_le.setMinimumSize(QtCore.QSize(100, 25))
        self._denom_ncycles_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._denom_ncycles_le.setFont(font)
        self._denom_ncycles_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._denom_ncycles_le.setAlignment(QtCore.Qt.AlignCenter)
        self._denom_ncycles_le.setObjectName("_denom_ncycles_le")
        self.verticalLayout.addWidget(self._denom_ncycles_le)
        self._njobs_le = QtWidgets.QLineEdit(self.groupBox)
        self._njobs_le.setMinimumSize(QtCore.QSize(100, 25))
        self._njobs_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._njobs_le.setFont(font)
        self._njobs_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._njobs_le.setAlignment(QtCore.Qt.AlignCenter)
        self._njobs_le.setObjectName("_njobs_le")
        self.verticalLayout.addWidget(self._njobs_le)
        self._log_freq_cbx = QtWidgets.QComboBox(self.groupBox)
        self._log_freq_cbx.setMinimumSize(QtCore.QSize(100, 25))
        self._log_freq_cbx.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._log_freq_cbx.setFont(font)
        self._log_freq_cbx.setObjectName("_log_freq_cbx")
        self._log_freq_cbx.addItem("")
        self._log_freq_cbx.addItem("")
        self.verticalLayout.addWidget(self._log_freq_cbx)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_8.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5.addWidget(self.groupBox)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout_9.addLayout(self.horizontalLayout_5)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_3.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout_3.setSpacing(10)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSpacing(10)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(15)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(15)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self._fig_chan_lb = QtWidgets.QLabel(self.groupBox_2)
        self._fig_chan_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_chan_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._fig_chan_lb.setFont(font)
        self._fig_chan_lb.setObjectName("_fig_chan_lb")
        self.verticalLayout_4.addWidget(self._fig_chan_lb)
        self._fig_baseline_lb = QtWidgets.QLabel(self.groupBox_2)
        self._fig_baseline_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_baseline_lb.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._fig_baseline_lb.setFont(font)
        self._fig_baseline_lb.setObjectName("_fig_baseline_lb")
        self.verticalLayout_4.addWidget(self._fig_baseline_lb)
        self._fig_freq_lb = QtWidgets.QLabel(self.groupBox_2)
        self._fig_freq_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_freq_lb.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._fig_freq_lb.setFont(font)
        self._fig_freq_lb.setObjectName("_fig_freq_lb")
        self.verticalLayout_4.addWidget(self._fig_freq_lb)
        self._fig_time_lb = QtWidgets.QLabel(self.groupBox_2)
        self._fig_time_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_time_lb.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._fig_time_lb.setFont(font)
        self._fig_time_lb.setObjectName("_fig_time_lb")
        self.verticalLayout_4.addWidget(self._fig_time_lb)
        self._fig_log_lb = QtWidgets.QLabel(self.groupBox_2)
        self._fig_log_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_log_lb.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self._fig_log_lb.setFont(font)
        self._fig_log_lb.setObjectName("_fig_log_lb")
        self.verticalLayout_4.addWidget(self._fig_log_lb)
        self.horizontalLayout_4.addLayout(self.verticalLayout_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(15)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self._fig_select_chan_btn = QtWidgets.QToolButton(self.groupBox_2)
        self._fig_select_chan_btn.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_select_chan_btn.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._fig_select_chan_btn.setFont(font)
        self._fig_select_chan_btn.setObjectName("_fig_select_chan_btn")
        self.verticalLayout_3.addWidget(self._fig_select_chan_btn)
        self._fig_baseline_le = QtWidgets.QLineEdit(self.groupBox_2)
        self._fig_baseline_le.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_baseline_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._fig_baseline_le.setFont(font)
        self._fig_baseline_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._fig_baseline_le.setAlignment(QtCore.Qt.AlignCenter)
        self._fig_baseline_le.setObjectName("_fig_baseline_le")
        self.verticalLayout_3.addWidget(self._fig_baseline_le)
        self._fig_freq_band_le = QtWidgets.QLineEdit(self.groupBox_2)
        self._fig_freq_band_le.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_freq_band_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._fig_freq_band_le.setFont(font)
        self._fig_freq_band_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._fig_freq_band_le.setAlignment(QtCore.Qt.AlignCenter)
        self._fig_freq_band_le.setObjectName("_fig_freq_band_le")
        self.verticalLayout_3.addWidget(self._fig_freq_band_le)
        self._fig_time_le = QtWidgets.QLineEdit(self.groupBox_2)
        self._fig_time_le.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_time_le.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._fig_time_le.setFont(font)
        self._fig_time_le.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._fig_time_le.setAlignment(QtCore.Qt.AlignCenter)
        self._fig_time_le.setObjectName("_fig_time_le")
        self.verticalLayout_3.addWidget(self._fig_time_le)
        self._fig_log_cbx = QtWidgets.QComboBox(self.groupBox_2)
        self._fig_log_cbx.setMinimumSize(QtCore.QSize(100, 25))
        self._fig_log_cbx.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._fig_log_cbx.setFont(font)
        self._fig_log_cbx.setObjectName("_fig_log_cbx")
        self._fig_log_cbx.addItem("")
        self._fig_log_cbx.addItem("")
        self.verticalLayout_3.addWidget(self._fig_log_cbx)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_4)
        self.line_4 = QtWidgets.QFrame(self.groupBox_2)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_6.addWidget(self.line_4)
        self._plot_btn = QtWidgets.QPushButton(self.groupBox_2)
        self._plot_btn.setMinimumSize(QtCore.QSize(120, 25))
        self._plot_btn.setMaximumSize(QtCore.QSize(300, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._plot_btn.setFont(font)
        self._plot_btn.setObjectName("_plot_btn")
        self.verticalLayout_6.addWidget(self._plot_btn)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.line = QtWidgets.QFrame(self.groupBox_2)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_3.addWidget(self.line)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSpacing(7)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self._baseline_correction_lb = QtWidgets.QLabel(self.groupBox_2)
        self._baseline_correction_lb.setMinimumSize(QtCore.QSize(100, 25))
        self._baseline_correction_lb.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self._baseline_correction_lb.setFont(font)
        self._baseline_correction_lb.setObjectName("_baseline_correction_lb")
        self.verticalLayout_5.addWidget(self._baseline_correction_lb)
        self.line_2 = QtWidgets.QFrame(self.groupBox_2)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_5.addWidget(self.line_2)
        self._zscore_btn = QtWidgets.QRadioButton(self.groupBox_2)
        self._zscore_btn.setMinimumSize(QtCore.QSize(0, 25))
        self._zscore_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._zscore_btn.setFont(font)
        self._zscore_btn.setChecked(True)
        self._zscore_btn.setObjectName("_zscore_btn")
        self.verticalLayout_5.addWidget(self._zscore_btn)
        self._zlogratio_btn = QtWidgets.QRadioButton(self.groupBox_2)
        self._zlogratio_btn.setMinimumSize(QtCore.QSize(0, 25))
        self._zlogratio_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._zlogratio_btn.setFont(font)
        self._zlogratio_btn.setObjectName("_zlogratio_btn")
        self.verticalLayout_5.addWidget(self._zlogratio_btn)
        self._mean_btn = QtWidgets.QRadioButton(self.groupBox_2)
        self._mean_btn.setMinimumSize(QtCore.QSize(0, 25))
        self._mean_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._mean_btn.setFont(font)
        self._mean_btn.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._mean_btn.setObjectName("_mean_btn")
        self.verticalLayout_5.addWidget(self._mean_btn)
        self._ratio_btn = QtWidgets.QRadioButton(self.groupBox_2)
        self._ratio_btn.setMinimumSize(QtCore.QSize(0, 25))
        self._ratio_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._ratio_btn.setFont(font)
        self._ratio_btn.setAccessibleDescription("")
        self._ratio_btn.setObjectName("_ratio_btn")
        self.verticalLayout_5.addWidget(self._ratio_btn)
        self._logratio_btn = QtWidgets.QRadioButton(self.groupBox_2)
        self._logratio_btn.setMinimumSize(QtCore.QSize(0, 25))
        self._logratio_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._logratio_btn.setFont(font)
        self._logratio_btn.setAccessibleDescription("")
        self._logratio_btn.setObjectName("_logratio_btn")
        self.verticalLayout_5.addWidget(self._logratio_btn)
        self._percent_btn = QtWidgets.QRadioButton(self.groupBox_2)
        self._percent_btn.setMinimumSize(QtCore.QSize(0, 25))
        self._percent_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self._percent_btn.setFont(font)
        self._percent_btn.setAccessibleDescription("")
        self._percent_btn.setObjectName("_percent_btn")
        self.verticalLayout_5.addWidget(self._percent_btn)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.verticalLayout_9.addWidget(self.groupBox_2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        MainWindow.setToolTip(_translate("MainWindow", "dividing by the mean of baseline values"))
        self.groupBox.setTitle(_translate("MainWindow", "Calculation Parameters"))
        self.Morlet_methodlb.setText(_translate("MainWindow", "Morlet Method"))
        self._compute_btn.setText(_translate("MainWindow", "Compute"))
        self._compute_btn.setShortcut(_translate("MainWindow", "Return"))
        self._chan_lb.setText(_translate("MainWindow", "Channels"))
        self._freq_band_lb.setText(_translate("MainWindow", "Frequency band (Hz)"))
        self._freq_step_lb.setText(_translate("MainWindow", "Frequency step (Hz)"))
        self._denom_ncycles_lb.setText(_translate("MainWindow", "Denominator of n_cycles for each freq"))
        self._njobs_lb.setText(_translate("MainWindow", "Number of jobs"))
        self._log_freq_lb.setText(_translate("MainWindow", "Log-spaced frequnecies"))
        self._select_chan_btn.setText(_translate("MainWindow", "..."))
        self._freq_step_le.setText(_translate("MainWindow", "1"))
        self._denom_ncycles_le.setText(_translate("MainWindow", "2"))
        self._njobs_le.setText(_translate("MainWindow", "3"))
        self._log_freq_cbx.setItemText(0, _translate("MainWindow", "False"))
        self._log_freq_cbx.setItemText(1, _translate("MainWindow", "True"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Figure Configuration"))
        self._fig_chan_lb.setText(_translate("MainWindow", "Channels"))
        self._fig_baseline_lb.setText(_translate("MainWindow", "Baseline"))
        self._fig_freq_lb.setText(_translate("MainWindow", "Frequency range (Hz)"))
        self._fig_time_lb.setText(_translate("MainWindow", "Time range (sec)"))
        self._fig_log_lb.setText(_translate("MainWindow", "Log transformation"))
        self._fig_select_chan_btn.setText(_translate("MainWindow", "..."))
        self._fig_baseline_le.setText(_translate("MainWindow", "0 0.1"))
        self._fig_log_cbx.setItemText(0, _translate("MainWindow", "False"))
        self._fig_log_cbx.setItemText(1, _translate("MainWindow", "True"))
        self._plot_btn.setText(_translate("MainWindow", "Plot"))
        self._baseline_correction_lb.setText(_translate("MainWindow", "Baseline Correction"))
        self._zscore_btn.setToolTip(_translate("MainWindow", "subtracting the mean of baseline values and \n"
"dividing by the standard deviation of baseline values"))
        self._zscore_btn.setText(_translate("MainWindow", "Z-score transformation (zscore)"))
        self._zlogratio_btn.setToolTip(_translate("MainWindow", "dividing by the mean of baseline values, \n"
"taking the log, and dividing by the standard deviation of log baseline values"))
        self._zlogratio_btn.setText(_translate("MainWindow", "Log z-score transformation (zlogratio)"))
        self._mean_btn.setToolTip(_translate("MainWindow", "subtracting the mean of baseline value"))
        self._mean_btn.setText(_translate("MainWindow", "Subtracting the mean (mean)"))
        self._ratio_btn.setToolTip(_translate("MainWindow", "dividing by the mean of baseline values"))
        self._ratio_btn.setText(_translate("MainWindow", "Dividing by the mean (ratio)"))
        self._logratio_btn.setToolTip(_translate("MainWindow", "dividing by the mean of baseline values and taking the log"))
        self._logratio_btn.setText(_translate("MainWindow", "Subtracting the mean and taking the log (logratio)"))
        self._percent_btn.setToolTip(_translate("MainWindow", "subtracting the mean of baseline values \n"
" followed by dividing by the mean of baseline values"))
        self._percent_btn.setText(_translate("MainWindow", "Subtracting the mean followed by dividing by the mean (percent)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())