# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 666)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1000, 600))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self._ieeg_viz_stack = QtWidgets.QStackedWidget(self.centralwidget)
        self._ieeg_viz_stack.setFocusPolicy(QtCore.Qt.NoFocus)
        self._ieeg_viz_stack.setStyleSheet("")
        self._ieeg_viz_stack.setObjectName("_ieeg_viz_stack")
        self.verticalLayout.addWidget(self._ieeg_viz_stack)
        MainWindow.setCentralWidget(self.centralwidget)
        self._toolbar = QtWidgets.QToolBar(MainWindow)
        self._toolbar.setObjectName("_toolbar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self._toolbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        self.menuFile.setFont(font)
        self.menuFile.setObjectName("menuFile")
        self.menuClear_Workbench = QtWidgets.QMenu(self.menuFile)
        self.menuClear_Workbench.setObjectName("menuClear_Workbench")
        self.menuExport_SEEG = QtWidgets.QMenu(self.menuFile)
        self.menuExport_SEEG.setObjectName("menuExport_SEEG")
        self.menuProcess = QtWidgets.QMenu(self.menubar)
        self.menuProcess.setObjectName("menuProcess")
        self.menuReference = QtWidgets.QMenu(self.menuProcess)
        self.menuReference.setObjectName("menuReference")
        self.menuDrop_bad = QtWidgets.QMenu(self.menuProcess)
        self.menuDrop_bad.setObjectName("menuDrop_bad")
        self.menuFilter = QtWidgets.QMenu(self.menuProcess)
        self.menuFilter.setObjectName("menuFilter")
        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
        self.menuEpilepsy = QtWidgets.QMenu(self.menuAnalysis)
        self.menuEpilepsy.setObjectName("menuEpilepsy")
        self.menuConnectivity = QtWidgets.QMenu(self.menuAnalysis)
        self.menuConnectivity.setObjectName("menuConnectivity")
        self.menuTime_Frequency_Response = QtWidgets.QMenu(self.menuAnalysis)
        self.menuTime_Frequency_Response.setObjectName("menuTime_Frequency_Response")
        self.menuPower_Spectral_Density = QtWidgets.QMenu(self.menuAnalysis)
        self.menuPower_Spectral_Density.setObjectName("menuPower_Spectral_Density")
        self.menuMNI_Transform = QtWidgets.QMenu(self.menuAnalysis)
        self.menuMNI_Transform.setObjectName("menuMNI_Transform")
        self.menuUnlinear = QtWidgets.QMenu(self.menuMNI_Transform)
        self.menuUnlinear.setObjectName("menuUnlinear")
        self.menuCross_Spectral_Density = QtWidgets.QMenu(self.menuAnalysis)
        self.menuCross_Spectral_Density.setObjectName("menuCross_Spectral_Density")
        self.menuAnatomical_labels = QtWidgets.QMenu(self.menuAnalysis)
        self.menuAnatomical_labels.setObjectName("menuAnatomical_labels")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuVisualization = QtWidgets.QMenu(self.menubar)
        self.menuVisualization.setObjectName("menuVisualization")
        self.menuLocalization = QtWidgets.QMenu(self.menubar)
        self.menuLocalization.setObjectName("menuLocalization")
        self.menuAlignment = QtWidgets.QMenu(self.menuLocalization)
        self.menuAlignment.setObjectName("menuAlignment")
        self.menuReconstruction = QtWidgets.QMenu(self.menuLocalization)
        self.menuReconstruction.setObjectName("menuReconstruction")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self._website_action = QtWidgets.QAction(MainWindow)
        self._website_action.setObjectName("_website_action")
        self._epileptogenic_index_action = QtWidgets.QAction(MainWindow)
        self._epileptogenic_index_action.setObjectName("_epileptogenic_index_action")
        self._high_frequency_action = QtWidgets.QAction(MainWindow)
        self._high_frequency_action.setObjectName("_high_frequency_action")
        self._nxn_coherence_action = QtWidgets.QAction(MainWindow)
        self._nxn_coherence_action.setObjectName("_nxn_coherence_action")
        self._resample_ieeg_action = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../icon/resample.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._resample_ieeg_action.setIcon(icon)
        self._resample_ieeg_action.setObjectName("_resample_ieeg_action")
        self.actionCrop = QtWidgets.QAction(MainWindow)
        self.actionCrop.setObjectName("actionCrop")
        self._load_t1_action = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../icon/mri.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._load_t1_action.setIcon(icon1)
        self._load_t1_action.setIconVisibleInMenu(True)
        self._load_t1_action.setShortcutVisibleInContextMenu(False)
        self._load_t1_action.setObjectName("_load_t1_action")
        self._load_ieeg_action = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../../icon/ieeg.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._load_ieeg_action.setIcon(icon2)
        self._load_ieeg_action.setObjectName("_load_ieeg_action")
        self._load_coordinates_action = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../../icon/coordinate.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._load_coordinates_action.setIcon(icon3)
        self._load_coordinates_action.setObjectName("_load_coordinates_action")
        self._clear_seeg_action = QtWidgets.QAction(MainWindow)
        self._clear_seeg_action.setObjectName("_clear_seeg_action")
        self._setting_action = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../../icon/subject.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._setting_action.setIcon(icon4)
        self._setting_action.setObjectName("_setting_action")
        self._parcellation_action = QtWidgets.QAction(MainWindow)
        self._parcellation_action.setObjectName("_parcellation_action")
        self.actionFreeSurfer = QtWidgets.QAction(MainWindow)
        self.actionFreeSurfer.setObjectName("actionFreeSurfer")
        self.actionFastSurfer = QtWidgets.QAction(MainWindow)
        self.actionFastSurfer.setObjectName("actionFastSurfer")
        self.actionDeepCSR = QtWidgets.QAction(MainWindow)
        self.actionDeepCSR.setObjectName("actionDeepCSR")
        self.actionAlign = QtWidgets.QAction(MainWindow)
        self.actionAlign.setObjectName("actionAlign")
        self.actionlocator = QtWidgets.QAction(MainWindow)
        self.actionlocator.setObjectName("actionlocator")
        self._monopolar_action = QtWidgets.QAction(MainWindow)
        self._monopolar_action.setObjectName("_monopolar_action")
        self._bipolar_action = QtWidgets.QAction(MainWindow)
        self._bipolar_action.setObjectName("_bipolar_action")
        self._average_action = QtWidgets.QAction(MainWindow)
        self._average_action.setObjectName("_average_action")
        self._load_ct_action = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../../icon/ct.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._load_ct_action.setIcon(icon5)
        self._load_ct_action.setObjectName("_load_ct_action")
        self._electrodes_action = QtWidgets.QAction(MainWindow)
        self._electrodes_action.setObjectName("_electrodes_action")
        self._rois_action = QtWidgets.QAction(MainWindow)
        self._rois_action.setObjectName("_rois_action")
        self._freeview_action = QtWidgets.QAction(MainWindow)
        self._freeview_action.setObjectName("_freeview_action")
        self._crop_ieeg_action = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("../../icon/scissor.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._crop_ieeg_action.setIcon(icon6)
        self._crop_ieeg_action.setObjectName("_crop_ieeg_action")
        self._drop_annotations_action = QtWidgets.QAction(MainWindow)
        self._drop_annotations_action.setObjectName("_drop_annotations_action")
        self._drop_white_matters_action = QtWidgets.QAction(MainWindow)
        self._drop_white_matters_action.setObjectName("_drop_white_matters_action")
        self._drop_gray_matters_action = QtWidgets.QAction(MainWindow)
        self._drop_gray_matters_action.setObjectName("_drop_gray_matters_action")
        self._generate_roi_signal_action = QtWidgets.QAction(MainWindow)
        self._generate_roi_signal_action.setObjectName("_generate_roi_signal_action")
        self._export_coordinates_action = QtWidgets.QAction(MainWindow)
        self._export_coordinates_action.setObjectName("_export_coordinates_action")
        self._display_mri_action = QtWidgets.QAction(MainWindow)
        self._display_mri_action.setObjectName("_display_mri_action")
        self._display_ct_action = QtWidgets.QAction(MainWindow)
        self._display_ct_action.setObjectName("_display_ct_action")
        self._plot_overlay_action = QtWidgets.QAction(MainWindow)
        self._plot_overlay_action.setObjectName("_plot_overlay_action")
        self.actionFreeSurfer_2 = QtWidgets.QAction(MainWindow)
        self.actionFreeSurfer_2.setObjectName("actionFreeSurfer_2")
        self.actionFastSurferr = QtWidgets.QAction(MainWindow)
        self.actionFastSurferr.setObjectName("actionFastSurferr")
        self.actionDeepCSR_2 = QtWidgets.QAction(MainWindow)
        self.actionDeepCSR_2.setObjectName("actionDeepCSR_2")
        self._conventional_align_action = QtWidgets.QAction(MainWindow)
        self._conventional_align_action.setObjectName("_conventional_align_action")
        self._export_anatomy_action = QtWidgets.QAction(MainWindow)
        self._export_anatomy_action.setObjectName("_export_anatomy_action")
        self._ants_align_action = QtWidgets.QAction(MainWindow)
        self._ants_align_action.setObjectName("_ants_align_action")
        self._recon_freesurfer_action = QtWidgets.QAction(MainWindow)
        self._recon_freesurfer_action.setObjectName("_recon_freesurfer_action")
        self._recon_fastsurfer_action = QtWidgets.QAction(MainWindow)
        self._recon_fastsurfer_action.setObjectName("_recon_fastsurfer_action")
        self._recon_deepcsr_action = QtWidgets.QAction(MainWindow)
        self._recon_deepcsr_action.setObjectName("_recon_deepcsr_action")
        self._ieeg_locator_action = QtWidgets.QAction(MainWindow)
        self._ieeg_locator_action.setObjectName("_ieeg_locator_action")
        self._psd_multitaper_action = QtWidgets.QAction(MainWindow)
        self._psd_multitaper_action.setObjectName("_psd_multitaper_action")
        self._psd_welch_action = QtWidgets.QAction(MainWindow)
        self._psd_welch_action.setObjectName("_psd_welch_action")
        self._tfr_multitaper_action = QtWidgets.QAction(MainWindow)
        self._tfr_multitaper_action.setObjectName("_tfr_multitaper_action")
        self._tfr_morlet_action = QtWidgets.QAction(MainWindow)
        self._tfr_morlet_action.setObjectName("_tfr_morlet_action")
        self._tfr_stockwell_action = QtWidgets.QAction(MainWindow)
        self._tfr_stockwell_action.setObjectName("_tfr_stockwell_action")
        self._nxn_coherency_action = QtWidgets.QAction(MainWindow)
        self._nxn_coherency_action.setObjectName("_nxn_coherency_action")
        self._github_action = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("../../icon/github.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._github_action.setIcon(icon7)
        self._github_action.setObjectName("_github_action")
        self._seeg_information = QtWidgets.QAction(MainWindow)
        self._seeg_information.setObjectName("_seeg_information")
        self._edit_channels_action = QtWidgets.QAction(MainWindow)
        self._edit_channels_action.setObjectName("_edit_channels_action")
        self._set_montage_action = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("../../icon/montage.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._set_montage_action.setIcon(icon8)
        self._set_montage_action.setObjectName("_set_montage_action")
        self._ieeg_info_action = QtWidgets.QAction(MainWindow)
        self._ieeg_info_action.setObjectName("_ieeg_info_action")
        self._channels_info_action = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("../../icon/electrodes.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._channels_info_action.setIcon(icon9)
        self._channels_info_action.setObjectName("_channels_info_action")
        self._mri_info_action = QtWidgets.QAction(MainWindow)
        self._mri_info_action.setObjectName("_mri_info_action")
        self._ct_info_action = QtWidgets.QAction(MainWindow)
        self._ct_info_action.setObjectName("_ct_info_action")
        self._screenshot_action = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("../../icon/screenshot.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._screenshot_action.setIcon(icon10)
        self._screenshot_action.setObjectName("_screenshot_action")
        self._ieeg_toolbar_action = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("../../icon/toolbar.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._ieeg_toolbar_action.setIcon(icon11)
        self._ieeg_toolbar_action.setObjectName("_ieeg_toolbar_action")
        self.actionClear_MRI = QtWidgets.QAction(MainWindow)
        self.actionClear_MRI.setObjectName("actionClear_MRI")
        self.actionClear_CT = QtWidgets.QAction(MainWindow)
        self.actionClear_CT.setObjectName("actionClear_CT")
        self._clear_coordinates_action = QtWidgets.QAction(MainWindow)
        self._clear_coordinates_action.setObjectName("_clear_coordinates_action")
        self._clear_mri_action = QtWidgets.QAction(MainWindow)
        self._clear_mri_action.setObjectName("_clear_mri_action")
        self._clear_ieeg_action = QtWidgets.QAction(MainWindow)
        self._clear_ieeg_action.setObjectName("_clear_ieeg_action")
        self._clear_ct_action = QtWidgets.QAction(MainWindow)
        self._clear_ct_action.setObjectName("_clear_ct_action")
        self._clear_coordinates_action_2 = QtWidgets.QAction(MainWindow)
        self._clear_coordinates_action_2.setObjectName("_clear_coordinates_action_2")
        self._clear_coordinate_action = QtWidgets.QAction(MainWindow)
        self._clear_coordinate_action.setObjectName("_clear_coordinate_action")
        self._acpc_alignment_action = QtWidgets.QAction(MainWindow)
        self._acpc_alignment_action.setObjectName("_acpc_alignment_action")
        self._epoch_action = QtWidgets.QAction(MainWindow)
        self._epoch_action.setObjectName("_epoch_action")
        self._fir_filter_action = QtWidgets.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("../../icon/filter.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self._fir_filter_action.setIcon(icon12)
        self._fir_filter_action.setObjectName("_fir_filter_action")
        self._iir_filter_action = QtWidgets.QAction(MainWindow)
        self._iir_filter_action.setObjectName("_iir_filter_action")
        self._export_set_action = QtWidgets.QAction(MainWindow)
        self._export_set_action.setObjectName("_export_set_action")
        self._export_edf_action = QtWidgets.QAction(MainWindow)
        self._export_edf_action.setObjectName("_export_edf_action")
        self._export_fif_action = QtWidgets.QAction(MainWindow)
        self._export_fif_action.setObjectName("_export_fif_action")
        self._load_anatomy_action = QtWidgets.QAction(MainWindow)
        self._load_anatomy_action.setObjectName("_load_anatomy_action")
        self.actionExport_Anatomy = QtWidgets.QAction(MainWindow)
        self.actionExport_Anatomy.setObjectName("actionExport_Anatomy")
        self._set_anatomy_action = QtWidgets.QAction(MainWindow)
        self._set_anatomy_action.setObjectName("_set_anatomy_action")
        self._compute_vep_atlas_action = QtWidgets.QAction(MainWindow)
        self._compute_vep_atlas_action.setObjectName("_compute_vep_atlas_action")
        self._linear_mni_action = QtWidgets.QAction(MainWindow)
        self._linear_mni_action.setObjectName("_linear_mni_action")
        self._custom_action = QtWidgets.QAction(MainWindow)
        self._custom_action.setObjectName("_custom_action")
        self._1xn_coherence_action = QtWidgets.QAction(MainWindow)
        self._1xn_coherence_action.setObjectName("_1xn_coherence_action")
        self._1xn_coherency_action = QtWidgets.QAction(MainWindow)
        self._1xn_coherency_action.setObjectName("_1xn_coherency_action")
        self._drop_unknown_matters_action = QtWidgets.QAction(MainWindow)
        self._drop_unknown_matters_action.setObjectName("_drop_unknown_matters_action")
        self._mni_fsaverage_action = QtWidgets.QAction(MainWindow)
        self._mni_fsaverage_action.setObjectName("_mni_fsaverage_action")
        self._mni_152_action = QtWidgets.QAction(MainWindow)
        self._mni_152_action.setObjectName("_mni_152_action")
        self._custom_action_2 = QtWidgets.QAction(MainWindow)
        self._custom_action_2.setObjectName("_custom_action_2")
        self._nx1_coherence_action = QtWidgets.QAction(MainWindow)
        self._nx1_coherence_action.setObjectName("_nx1_coherence_action")
        self._nx1__coherency_action = QtWidgets.QAction(MainWindow)
        self._nx1__coherency_action.setObjectName("_nx1__coherency_action")
        self._csd_fourier_action = QtWidgets.QAction(MainWindow)
        self._csd_fourier_action.setObjectName("_csd_fourier_action")
        self._csd_morlet_action = QtWidgets.QAction(MainWindow)
        self._csd_morlet_action.setObjectName("_csd_morlet_action")
        self._csd_multitaper_action = QtWidgets.QAction(MainWindow)
        self._csd_multitaper_action.setObjectName("_csd_multitaper_action")
        self._a2009s_action = QtWidgets.QAction(MainWindow)
        self._a2009s_action.setObjectName("_a2009s_action")
        self._dkt_action = QtWidgets.QAction(MainWindow)
        self._dkt_action.setObjectName("_dkt_action")
        self._aseg_action = QtWidgets.QAction(MainWindow)
        self._aseg_action.setObjectName("_aseg_action")
        self._vep_action = QtWidgets.QAction(MainWindow)
        self._vep_action.setObjectName("_vep_action")
        self._toolbar.addAction(self._load_t1_action)
        self._toolbar.addAction(self._load_ct_action)
        self._toolbar.addAction(self._load_ieeg_action)
        self._toolbar.addAction(self._load_coordinates_action)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._crop_ieeg_action)
        self._toolbar.addAction(self._resample_ieeg_action)
        self._toolbar.addAction(self._fir_filter_action)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._channels_info_action)
        self._toolbar.addAction(self._set_montage_action)
        self._toolbar.addAction(self._setting_action)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._screenshot_action)
        self._toolbar.addAction(self._ieeg_toolbar_action)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._github_action)
        self.menuClear_Workbench.addAction(self._clear_mri_action)
        self.menuClear_Workbench.addAction(self._clear_ct_action)
        self.menuClear_Workbench.addAction(self._clear_ieeg_action)
        self.menuClear_Workbench.addAction(self._clear_coordinate_action)
        self.menuExport_SEEG.addAction(self._export_fif_action)
        self.menuExport_SEEG.addAction(self._export_edf_action)
        self.menuExport_SEEG.addAction(self._export_set_action)
        self.menuFile.addAction(self._load_t1_action)
        self.menuFile.addAction(self._load_ct_action)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self._load_ieeg_action)
        self.menuFile.addAction(self.menuExport_SEEG.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self._load_coordinates_action)
        self.menuFile.addAction(self._export_coordinates_action)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.menuClear_Workbench.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self._setting_action)
        self.menuReference.addAction(self._monopolar_action)
        self.menuReference.addAction(self._bipolar_action)
        self.menuReference.addAction(self._average_action)
        self.menuDrop_bad.addAction(self._drop_annotations_action)
        self.menuDrop_bad.addAction(self._drop_white_matters_action)
        self.menuDrop_bad.addAction(self._drop_gray_matters_action)
        self.menuDrop_bad.addAction(self._drop_unknown_matters_action)
        self.menuFilter.addAction(self._fir_filter_action)
        self.menuFilter.addAction(self._iir_filter_action)
        self.menuProcess.addAction(self._set_montage_action)
        self.menuProcess.addSeparator()
        self.menuProcess.addAction(self._crop_ieeg_action)
        self.menuProcess.addAction(self._resample_ieeg_action)
        self.menuProcess.addAction(self.menuFilter.menuAction())
        self.menuProcess.addAction(self.menuReference.menuAction())
        self.menuProcess.addAction(self.menuDrop_bad.menuAction())
        self.menuProcess.addSeparator()
        self.menuProcess.addAction(self._epoch_action)
        self.menuProcess.addSeparator()
        self.menuProcess.addAction(self._generate_roi_signal_action)
        self.menuEpilepsy.addAction(self._epileptogenic_index_action)
        self.menuEpilepsy.addAction(self._high_frequency_action)
        self.menuConnectivity.addAction(self._nx1_coherence_action)
        self.menuConnectivity.addAction(self._nxn_coherence_action)
        self.menuConnectivity.addSeparator()
        self.menuConnectivity.addAction(self._nx1__coherency_action)
        self.menuConnectivity.addAction(self._nxn_coherency_action)
        self.menuTime_Frequency_Response.addAction(self._tfr_multitaper_action)
        self.menuTime_Frequency_Response.addAction(self._tfr_morlet_action)
        self.menuTime_Frequency_Response.addAction(self._tfr_stockwell_action)
        self.menuPower_Spectral_Density.addAction(self._psd_multitaper_action)
        self.menuPower_Spectral_Density.addAction(self._psd_welch_action)
        self.menuUnlinear.addAction(self._mni_fsaverage_action)
        self.menuUnlinear.addAction(self._mni_152_action)
        self.menuUnlinear.addAction(self._custom_action_2)
        self.menuMNI_Transform.addAction(self._linear_mni_action)
        self.menuMNI_Transform.addAction(self.menuUnlinear.menuAction())
        self.menuCross_Spectral_Density.addAction(self._csd_fourier_action)
        self.menuCross_Spectral_Density.addAction(self._csd_morlet_action)
        self.menuCross_Spectral_Density.addAction(self._csd_multitaper_action)
        self.menuAnatomical_labels.addAction(self._vep_action)
        self.menuAnatomical_labels.addAction(self._a2009s_action)
        self.menuAnatomical_labels.addAction(self._dkt_action)
        self.menuAnatomical_labels.addAction(self._aseg_action)
        self.menuAnalysis.addAction(self.menuAnatomical_labels.menuAction())
        self.menuAnalysis.addAction(self.menuMNI_Transform.menuAction())
        self.menuAnalysis.addSeparator()
        self.menuAnalysis.addAction(self.menuPower_Spectral_Density.menuAction())
        self.menuAnalysis.addAction(self.menuCross_Spectral_Density.menuAction())
        self.menuAnalysis.addAction(self.menuTime_Frequency_Response.menuAction())
        self.menuAnalysis.addSeparator()
        self.menuAnalysis.addAction(self.menuConnectivity.menuAction())
        self.menuAnalysis.addSeparator()
        self.menuAnalysis.addAction(self.menuEpilepsy.menuAction())
        self.menuHelp.addAction(self._github_action)
        self.menuVisualization.addAction(self._parcellation_action)
        self.menuVisualization.addSeparator()
        self.menuVisualization.addAction(self._electrodes_action)
        self.menuVisualization.addAction(self._rois_action)
        self.menuVisualization.addSeparator()
        self.menuVisualization.addAction(self._freeview_action)
        self.menuAlignment.addAction(self._conventional_align_action)
        self.menuAlignment.addAction(self._ants_align_action)
        self.menuReconstruction.addAction(self._recon_freesurfer_action)
        self.menuReconstruction.addAction(self._recon_fastsurfer_action)
        self.menuReconstruction.addAction(self._recon_deepcsr_action)
        self.menuLocalization.addAction(self._acpc_alignment_action)
        self.menuLocalization.addAction(self.menuAlignment.menuAction())
        self.menuLocalization.addAction(self._plot_overlay_action)
        self.menuLocalization.addSeparator()
        self.menuLocalization.addAction(self.menuReconstruction.menuAction())
        self.menuLocalization.addAction(self._compute_vep_atlas_action)
        self.menuLocalization.addSeparator()
        self.menuLocalization.addAction(self._edit_channels_action)
        self.menuLocalization.addAction(self._ieeg_locator_action)
        self.menuView.addAction(self._display_mri_action)
        self.menuView.addAction(self._display_ct_action)
        self.menuView.addSeparator()
        self.menuView.addAction(self._mri_info_action)
        self.menuView.addAction(self._ct_info_action)
        self.menuView.addAction(self._ieeg_info_action)
        self.menuView.addAction(self._channels_info_action)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuLocalization.menuAction())
        self.menubar.addAction(self.menuProcess.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuVisualization.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self._toolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuClear_Workbench.setTitle(_translate("MainWindow", "Clear Workbench"))
        self.menuExport_SEEG.setTitle(_translate("MainWindow", "Export SEEG"))
        self.menuProcess.setTitle(_translate("MainWindow", "Signal"))
        self.menuReference.setTitle(_translate("MainWindow", "Reference"))
        self.menuDrop_bad.setTitle(_translate("MainWindow", "Drop channels"))
        self.menuFilter.setTitle(_translate("MainWindow", "Filter"))
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))
        self.menuEpilepsy.setTitle(_translate("MainWindow", "Epilepsy"))
        self.menuConnectivity.setTitle(_translate("MainWindow", "Spectral Connectivity"))
        self.menuTime_Frequency_Response.setTitle(_translate("MainWindow", "Time-Frequency Response"))
        self.menuPower_Spectral_Density.setTitle(_translate("MainWindow", "Power Spectral Density"))
        self.menuMNI_Transform.setTitle(_translate("MainWindow", "MNI Transform"))
        self.menuUnlinear.setTitle(_translate("MainWindow", "Unlinear"))
        self.menuCross_Spectral_Density.setTitle(_translate("MainWindow", "Cross Spectral Density"))
        self.menuAnatomical_labels.setTitle(_translate("MainWindow", "Anatomical labels"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuVisualization.setTitle(_translate("MainWindow", "Visualization"))
        self.menuLocalization.setTitle(_translate("MainWindow", "Localization "))
        self.menuAlignment.setTitle(_translate("MainWindow", "Alignment"))
        self.menuReconstruction.setTitle(_translate("MainWindow", "Reconstruction"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self._website_action.setText(_translate("MainWindow", "Website"))
        self._epileptogenic_index_action.setText(_translate("MainWindow", "Epileptogenic Index"))
        self._high_frequency_action.setText(_translate("MainWindow", "High Frequency Oscillations"))
        self._nxn_coherence_action.setText(_translate("MainWindow", "NxN Coherence"))
        self._resample_ieeg_action.setText(_translate("MainWindow", "Resample"))
        self.actionCrop.setText(_translate("MainWindow", "Crop"))
        self._load_t1_action.setText(_translate("MainWindow", "Load MRI"))
        self._load_t1_action.setShortcut(_translate("MainWindow", "Ctrl+T"))
        self._load_ieeg_action.setText(_translate("MainWindow", "Load SEEG"))
        self._load_ieeg_action.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self._load_coordinates_action.setText(_translate("MainWindow", "Load Coordinates"))
        self._load_coordinates_action.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self._clear_seeg_action.setText(_translate("MainWindow", "Clear SEEG"))
        self._setting_action.setText(_translate("MainWindow", "Setting..."))
        self._parcellation_action.setText(_translate("MainWindow", "Parcellation"))
        self.actionFreeSurfer.setText(_translate("MainWindow", "FreeSurfer"))
        self.actionFastSurfer.setText(_translate("MainWindow", "FastSurfer"))
        self.actionDeepCSR.setText(_translate("MainWindow", "DeepCSR"))
        self.actionAlign.setText(_translate("MainWindow", "Align"))
        self.actionlocator.setText(_translate("MainWindow", "Locator"))
        self._monopolar_action.setText(_translate("MainWindow", "Monopolar"))
        self._bipolar_action.setText(_translate("MainWindow", "Bipolar"))
        self._average_action.setText(_translate("MainWindow", "Average"))
        self._load_ct_action.setText(_translate("MainWindow", "Load CT"))
        self._load_ct_action.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self._electrodes_action.setText(_translate("MainWindow", "Electrodes with ROIs"))
        self._rois_action.setText(_translate("MainWindow", "Regions of Interests"))
        self._freeview_action.setText(_translate("MainWindow", "Freeview"))
        self._crop_ieeg_action.setText(_translate("MainWindow", "Crop"))
        self._drop_annotations_action.setText(_translate("MainWindow", "Annotations"))
        self._drop_white_matters_action.setText(_translate("MainWindow", "White matters"))
        self._drop_gray_matters_action.setText(_translate("MainWindow", "Gray matters"))
        self._generate_roi_signal_action.setText(_translate("MainWindow", "Generate ROI signals by averaging"))
        self._export_coordinates_action.setText(_translate("MainWindow", "Export Coordinates"))
        self._display_mri_action.setText(_translate("MainWindow", "Display MRI"))
        self._display_ct_action.setText(_translate("MainWindow", "Display CT"))
        self._plot_overlay_action.setText(_translate("MainWindow", "Overlay CT on MRI"))
        self.actionFreeSurfer_2.setText(_translate("MainWindow", "FreeSurfer"))
        self.actionFastSurferr.setText(_translate("MainWindow", "FastSurferr"))
        self.actionDeepCSR_2.setText(_translate("MainWindow", "DeepCSR"))
        self._conventional_align_action.setText(_translate("MainWindow", "Dipy"))
        self._export_anatomy_action.setText(_translate("MainWindow", "Export Anatomy"))
        self._ants_align_action.setText(_translate("MainWindow", "AntsPy"))
        self._recon_freesurfer_action.setText(_translate("MainWindow", "FreeSurfer"))
        self._recon_fastsurfer_action.setText(_translate("MainWindow", "FastSurfer"))
        self._recon_deepcsr_action.setText(_translate("MainWindow", "DeepCSR"))
        self._ieeg_locator_action.setText(_translate("MainWindow", "iEEG Locator"))
        self._psd_multitaper_action.setText(_translate("MainWindow", "Multitaper"))
        self._psd_welch_action.setText(_translate("MainWindow", "Welch"))
        self._tfr_multitaper_action.setText(_translate("MainWindow", "Multitaper"))
        self._tfr_morlet_action.setText(_translate("MainWindow", "Morlet Wavelets"))
        self._tfr_stockwell_action.setText(_translate("MainWindow", "Stockwell"))
        self._nxn_coherency_action.setText(_translate("MainWindow", "NxN Coherency"))
        self._github_action.setText(_translate("MainWindow", "Github"))
        self._seeg_information.setText(_translate("MainWindow", "Signal Information"))
        self._edit_channels_action.setText(_translate("MainWindow", "Edit Channels"))
        self._set_montage_action.setText(_translate("MainWindow", "Set Montage"))
        self._ieeg_info_action.setText(_translate("MainWindow", "iEEG Information"))
        self._channels_info_action.setText(_translate("MainWindow", "Channels Information"))
        self._mri_info_action.setText(_translate("MainWindow", "MRI Information"))
        self._ct_info_action.setText(_translate("MainWindow", "CT Information"))
        self._screenshot_action.setText(_translate("MainWindow", "Screenshot"))
        self._ieeg_toolbar_action.setText(_translate("MainWindow", "Toolbar"))
        self.actionClear_MRI.setText(_translate("MainWindow", "Clear MRI"))
        self.actionClear_CT.setText(_translate("MainWindow", "Clear CT"))
        self._clear_coordinates_action.setText(_translate("MainWindow", "Clear Coordinates"))
        self._clear_mri_action.setText(_translate("MainWindow", "Clear MRI"))
        self._clear_ieeg_action.setText(_translate("MainWindow", "Clear SEEG"))
        self._clear_ct_action.setText(_translate("MainWindow", "Clear CT"))
        self._clear_coordinates_action_2.setText(_translate("MainWindow", "Clear Coordinates"))
        self._clear_coordinate_action.setText(_translate("MainWindow", "Clear Coordinates"))
        self._acpc_alignment_action.setText(_translate("MainWindow", "ACPC Alignment"))
        self._epoch_action.setText(_translate("MainWindow", "Epoch"))
        self._fir_filter_action.setText(_translate("MainWindow", "FIR filter"))
        self._iir_filter_action.setText(_translate("MainWindow", "IIR filter"))
        self._export_set_action.setText(_translate("MainWindow", "EEGLAB (.set)"))
        self._export_edf_action.setText(_translate("MainWindow", "EDF+ (.edf)"))
        self._export_fif_action.setText(_translate("MainWindow", "Neuromag (.fif)"))
        self._load_anatomy_action.setText(_translate("MainWindow", "Load Anatomy"))
        self.actionExport_Anatomy.setText(_translate("MainWindow", "Export Anatomy"))
        self._set_anatomy_action.setText(_translate("MainWindow", "Set Anatomy"))
        self._compute_vep_atlas_action.setText(_translate("MainWindow", "Compute VEP atlas"))
        self._linear_mni_action.setText(_translate("MainWindow", "Linear"))
        self._custom_action.setText(_translate("MainWindow", "Custom..."))
        self._1xn_coherence_action.setText(_translate("MainWindow", "Coherence"))
        self._1xn_coherency_action.setText(_translate("MainWindow", "Coherency"))
        self._drop_unknown_matters_action.setText(_translate("MainWindow", "Unknown"))
        self._mni_fsaverage_action.setText(_translate("MainWindow", "MNI305 (fsaverage)"))
        self._mni_152_action.setText(_translate("MainWindow", "MNI152"))
        self._custom_action_2.setText(_translate("MainWindow", "Custom ..."))
        self._nx1_coherence_action.setText(_translate("MainWindow", "Nx1 Coherence"))
        self._nx1__coherency_action.setText(_translate("MainWindow", "Nx1 Coherency"))
        self._csd_fourier_action.setText(_translate("MainWindow", "Fourier"))
        self._csd_morlet_action.setText(_translate("MainWindow", "Morlet"))
        self._csd_multitaper_action.setText(_translate("MainWindow", "Multitaper"))
        self._a2009s_action.setText(_translate("MainWindow", "aparc.a2009s+aseg"))
        self._dkt_action.setText(_translate("MainWindow", "aparc.DKTatlas+aseg"))
        self._aseg_action.setText(_translate("MainWindow", "aparc+aseg"))
        self._vep_action.setText(_translate("MainWindow", "aparc+aseg.vep"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
