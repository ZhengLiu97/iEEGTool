# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：main_win.py
@Author  ：Barry
@Date    ：2022/2/18 1:39 
"""
import os
import gc
import glob
import platform

import mne
import matplotlib
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib

from mne.transforms import apply_trans
from mne.io import BaseRaw
from matplotlib import pyplot as plt
from dipy.align import resample
from collections import OrderedDict
from nibabel.viewers import OrthoSlicer3D
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyleFactory, QDesktopWidget, \
                            QFileDialog, QMessageBox, QShortcut, QAction
from PyQt5.QtCore import pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices, QKeySequence, QFont, QPixmap

from gui.main_ui import Ui_MainWindow
from gui.resample_win import ResampleWin
from gui.crop_win import CropWin
from gui.ieeg_info_win import iEEGInfoWin
from gui.info_win import InfoWin
from gui.list_win import ItemSelectionWin
from gui.electrodes_viz_win import ElectrodesWin
from gui.rois_viz_win import ROIsWin
from gui.fir_filter_win import FIRFilterWin
from gui.compute_ei_win import EIWin
from gui.compute_hfo_win import RMSHFOWin
from gui.table_win import TableWin
from gui.psd_multitaper_win import MultitaperPSDWin
from gui.psd_welch_win import WelchPSDWin
from gui.csd_fourier_win import FourierCSDWin
from gui.csd_morlet_win import MorletCSDWin
from gui.csd_multitaper_win import MultitaperCSDWin
from gui.tfr_multitaper_win import TFRMultitaperWin
from gui.tfr_morlet_win import TFRMorletWin
from gui.NxN_connectivity_win import NxNSpectraConWin
from gui.Nx1_connectivity_win import Nx1SpectraConWin
from utils.subject import Subject
from utils.thread import *
from utils.log_config import create_logger
from utils.decorator import safe_event
from utils.electrodes import Electrodes
from utils.contacts import calc_ch_pos, calc_bipolar_chs_pos, reorder_chs, reorder_chs_df
from utils.process import get_chan_group, set_montage, clean_chans, get_montage, mne_bipolar, \
                          set_laplacian_ref
from utils.get_anatomical_labels import labelling_contacts_vol_fs_mgz
from viz.locate_ieeg import locate_ieeg

matplotlib.use('Qt5Agg')
mne.viz.set_browser_backend('pyqtgraph')

SYSTEM = platform.system()

logger = create_logger(filename='iEEGTool.log')

default_path = 'data'
freesurfer_path = 'data/freesurfer'


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('iEEG Tool')
        self.setWindowIcon(QIcon('icon/iEEGTool.ico'))

        self._center_win()
        self._slot_connection()
        self._short_cut()
        self._set_icon()

        self.ieeg_title = ''
        self.mri_title = ''
        self.ct_title = ''

        # self.subject = Subject('sample')
        self.subject = Subject('cenjianv')
        self._info = {'subject_name': '', 'age': '', 'gender': ''}
        self.subjects_dir = freesurfer_path
        self.chs_name = ''

        self.electrodes = Electrodes()
        self.wm_chs = list()
        self.gm_chs = list()
        self.unknown_chs = list()
        self.set_montage = False
        self.mri_path = ''

        self.wins = dict(crop_win=None, resample_win=None, fir_filter_win=None, iir_filter_win=None,
                         psd_multitaper_win=None, psd_welch_win=None, csd_fourier_win=None, csd_morlet_win=None,
                         csd_multitaper_win=None, tfr_multitaper_win=None, tfr_morlet_win=None, nxn_con_win=None,
                         nx1_con_win=None, ei_win=None, hfo_win=None, elec_viz_win=None, rois_viz_win=None,
                         ieeg_info_win=None, info_win=None, table_win=None, monopolar_win=None)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _slot_connection(self):
        # File Menu
        self._load_t1_action.triggered.connect(self._import_t1)
        self._load_ct_action.triggered.connect(self._import_ct)
        self._load_ieeg_action.triggered.connect(self._import_ieeg)
        self._load_coordinates_action.triggered.connect(self._import_coord)
        self._export_fif_action.triggered.connect(self._export_ieeg_fif)
        self._export_edf_action.triggered.connect(self._export_ieeg_edf)
        self._export_set_action.triggered.connect(self._export_ieeg_set)
        self._export_coordinates_action.triggered.connect(self._export_coords)

        self._clear_mri_action.triggered.connect(self._clear_mri)
        self._clear_ct_action.triggered.connect(self._clear_ct)
        self._clear_coordinate_action.triggered.connect(self._clear_coordinates)
        self._clear_ieeg_action.triggered.connect(self._clear_ieeg)

        self._setting_action.triggered.connect(self._write_info)

        # View Menu
        self._ieeg_info_action.triggered.connect(self._view_ieeg_info)
        self._channels_info_action.triggered.connect(self._view_chs_info)

        # Localization Menu
        self._display_mri_action.triggered.connect(self._display_t1)
        self._display_ct_action.triggered.connect(self._display_ct)
        self._plot_overlay_action.triggered.connect(self._plot_overlay)
        self._ieeg_locator_action.triggered.connect(self._locate_ieeg)

        # Signal Menu
        self._set_montage_action.triggered.connect(self._set_ieeg_montage)
        self._crop_ieeg_action.triggered.connect(self._crop_ieeg)
        self._resample_ieeg_action.triggered.connect(self._resample_ieeg)
        self._fir_filter_action.triggered.connect(self._fir_filter_ieeg)
        self._monopolar_action.triggered.connect(self._monopolar_reference)
        self._bipolar_action.triggered.connect(self._bipolar_ieeg)
        self._laplacian_action.triggered.connect(self._laplacian_ieeg)
        self._average_action.triggered.connect(self._average_reference)
        self._drop_annotations_action.triggered.connect(self._drop_bad_from_annotations)
        self._drop_white_matters_action.triggered.connect(self._drop_wm_chs)
        self._drop_gray_matters_action.triggered.connect(self._drop_gm_chs)
        self._drop_unknown_matters_action.triggered.connect(self._drop_unknown_chs)

        # Analysis Menu
        self._anatomical_labeling_action.triggered.connect(self._set_anatomy)
        self._psd_multitaper_action.triggered.connect(self._psd_multitaper)
        self._psd_welch_action.triggered.connect(self._psd_welch)
        self._csd_fourier_action.triggered.connect(self._csd_fourier)
        self._csd_morlet_action.triggered.connect(self._csd_morlet)
        self._csd_multitaper_action.triggered.connect(self._csd_multitaper)
        self._tfr_multitaper_action.triggered.connect(self._tfr_multitaper)
        self._tfr_morlet_action.triggered.connect(self._tfr_morlet)
        self._nxn_coherence_action.triggered.connect(lambda: self._nxn_con('coh'))
        self._nx1_coherence_action.triggered.connect(lambda: self._nx1_con('coh'))
        self._epileptogenic_index_action.triggered.connect(self._compute_ei)
        self._high_frequency_action.triggered.connect(self._compute_hfo)

        # Visualization Menu
        self._electrodes_action.triggered.connect(self._electrodes_viz)
        self._rois_action.triggered.connect(self._rois_viz)
        self._freeview_action.triggered.connect(self._open_freeview)

        # Help Menu
        self._github_action.triggered.connect(self._open_github)

        # Toolbar
        self._screenshot_action.triggered.connect(self._take_screenshot)
        self._ieeg_toolbar_action.triggered.connect(self._viz_ieeg_toolbar)

    def _update_fig(self):
        raw = self.subject.get_ieeg()
        # fig = mne.viz.plot_raw(raw, remove_dc=True, color='k',
        #                        n_channels=30, scalings='auto', show=False)
        fig = mne.viz.plot_raw(raw, remove_dc=True, color='k',
                               n_channels=30, scalings={'seeg': 1e-4}, show=False)
        fig.mne.overview_bar.setVisible(False)
        remove_actions = ['SSP', 'Settings']
        for action in fig.mne.toolbar.actions():
            if action.text() in remove_actions:
                fig.mne.toolbar.removeAction(action)

        # Set the default selection of the overview_menu is Hidden
        for index, action in enumerate(fig.mne.overview_menu.actions()):
            # the action is an instance of QWidgetAction
            # in MNE they set its defaultWidget as a radioWidget
            # here we find the index of hidden is 3
            # so we set the defaultWidget of QWidgetAction when index is 3 Checked
            if index == 3:
                action.defaultWidget().setChecked(True)
                # print(action.defaultWidget().text())
                fig._overview_mode_changed('hidden')
        fig.statusBar().setVisible(False)
        fig._set_annotations_visible(False)
        fig.mne.toolbar.setVisible(False)
        # print(type(fig))

        if self._ieeg_viz_stack.count() > 0:
            widget = self._ieeg_viz_stack.widget(0)
            self._ieeg_viz_stack.removeWidget(widget)
        self._ieeg_viz_stack.addWidget(fig)

    def _short_cut(self):
        QShortcut(QKeySequence(self.tr("F10")), self, self.showNormal)
        QShortcut(QKeySequence(self.tr("F11")), self, self.showMaximized)
        QShortcut(QKeySequence(self.tr("Ctrl+S")), self, self._export_ieeg_fif)
        QShortcut(QKeySequence(self.tr("Ctrl+Q")), self, self.close)

    def _set_icon(self):
        mri_icon = QIcon()
        mri_icon.addPixmap(QPixmap("icon/mri.svg"), QIcon.Normal, QIcon.Off)
        self._load_t1_action.setIcon(mri_icon)

        ct_icon = QIcon()
        ct_icon.addPixmap(QPixmap("icon/ct.svg"), QIcon.Normal, QIcon.Off)
        self._load_ct_action.setIcon(ct_icon)

        ieeg_icon = QIcon()
        ieeg_icon.addPixmap(QPixmap("icon/ieeg.svg"), QIcon.Normal, QIcon.Off)
        self._load_ieeg_action.setIcon(ieeg_icon)

        coord_icon = QIcon()
        coord_icon.addPixmap(QPixmap("icon/coordinate.svg"), QIcon.Normal, QIcon.Off)
        self._load_coordinates_action.setIcon(coord_icon)

        crop_icon = QIcon()
        crop_icon.addPixmap(QPixmap("icon/scissor.svg"), QIcon.Normal, QIcon.Off)
        self._crop_ieeg_action.setIcon(crop_icon)

        resample_icon = QIcon()
        resample_icon.addPixmap(QPixmap("icon/resample.svg"), QIcon.Normal, QIcon.Off)
        self._resample_ieeg_action.setIcon(resample_icon)

        filter_icon = QIcon()
        filter_icon.addPixmap(QPixmap("icon/filter.svg"), QIcon.Normal, QIcon.Off)
        self._fir_filter_action.setIcon(filter_icon)

        elec_icon = QIcon()
        elec_icon.addPixmap(QPixmap("icon/electrodes.svg"), QIcon.Normal, QIcon.Off)
        self._channels_info_action.setIcon(elec_icon)

        montage_icon = QIcon()
        montage_icon.addPixmap(QPixmap("icon/montage.svg"), QIcon.Normal, QIcon.Off)
        self._set_montage_action.setIcon(montage_icon)

        setting_icon = QIcon()
        setting_icon.addPixmap(QPixmap("icon/subject.svg"), QIcon.Normal, QIcon.Off)
        self._setting_action.setIcon(setting_icon)

        github_icon = QIcon()
        github_icon.addPixmap(QPixmap("icon/github.svg"), QIcon.Normal, QIcon.Off)
        self._github_action.setIcon(github_icon)

        screenshot_icon = QIcon()
        screenshot_icon.addPixmap(QPixmap("icon/screenshot.svg"), QIcon.Normal, QIcon.Off)
        self._screenshot_action.setIcon(screenshot_icon)

        ieeg_toolbar_icon = QIcon()
        ieeg_toolbar_icon.addPixmap(QPixmap("icon/toolbar.svg"), QIcon.Normal, QIcon.Off)
        self._ieeg_toolbar_action.setIcon(ieeg_toolbar_icon)

    ######################################################################
    #                                Slot                                #
    ######################################################################
    # File Menu
    def _import_t1(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Load T1 MRI", default_path,
                                                  filter="MRI (*.nii *.nii.gz *.mgz)")
        if len(fpath):
            t1 = nib.load(fpath)
            self.subject.set_t1(t1)
            self.mri_title = f'MRI {fpath}'
            self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                                f'{self.ct_title}')
            QMessageBox.information(self, 'T1-MRI', 'T1-MRI Loaded')

    def _import_ct(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Load CT", default_path,
                                                  filter="CT (*.nii *.nii.gz *.mgz)")
        if len(fpath):
            ct = nib.load(fpath)
            self.subject.set_ct(ct)
            self.ct_title = f'CT {fpath}'
            self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                                f'{self.ct_title}')
            QMessageBox.information(self, 'CT', 'CT Loaded')

    def _import_ieeg(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Import iEEG", default_path,
                                               filter="iEEG (*.edf *.set *.fif *.vhdr)")
        self.ieeg_title = f"iEEG {fpath}"
        if len(fpath):
            logger.info(f'iEEG path {fpath}')
            self._import_ieeg_thread = ImportSEEG(fpath)
            self._import_ieeg_thread.LOAD_SIGNAL.connect(self._get_ieeg)
            self._import_ieeg_thread.start()

    def _get_ieeg(self, ieeg):
        ieeg = clean_chans(ieeg)
        clean_chans(ieeg)
        logger.info(f'Cleaning channels finished!')
        try:
            chs = reorder_chs(ieeg.ch_names)
        except:
            chs = None
        if chs is not None:
            ieeg.reorder_channels(chs)
        self.subject.remove_ieeg()
        self.subject.set_ieeg(ieeg)
        self.subject.set_electrodes(None)
        self._update_fig()
        self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                            f'{self.ct_title}')
        logger.info('Set iEEG')

    def _import_coord(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Coordinates', default_path,
                                               filter='Coordinates (*.txt *.tsv)')
        if len(fname):
            # try:
            self.chs_name = fname
            basename = os.path.basename(fname)
            self.subject.set_name(basename[:basename.rfind('.txt')])
            coords_df = pd.read_table(fname)
            df = reorder_chs_df(coords_df)
            if df is not None:
                coords_df = df
            ch_names = coords_df['Channel'].to_list()
            x = coords_df['x'].to_numpy()
            y = coords_df['y'].to_numpy()
            z = coords_df['z'].to_numpy()

            self.electrodes.clean()
            self.electrodes.set_ch_names(ch_names)
            self.electrodes.set_ch_xyz([x, y, z])

            if 'ROI' in coords_df.columns:
                rois = coords_df['ROI'].to_list()
                # print(rois)
                self.electrodes.set_issues(rois)
                self.electrodes.set_anatomy('ROI', rois)
            self.set_montage = False
            logger.info("Importing channels' coordinates finished!")
            QMessageBox.information(self, 'Coordinates', "Importing channels' coordinates finished!")
            # except:
            #     QMessageBox.warning(self, 'Coordinates', 'Wrong file format!')

    def _export_ieeg_fif(self):
        ieeg_format = '.fif'
        default_fname = os.path.join(default_path, self.subject.get_name() + ieeg_format)
        fname, _ = QFileDialog.getSaveFileName(self, 'iEEG', default_fname,
                                               filter=f"Neuromag (*{ieeg_format})")
        if len(fname):
            raw = self.subject.get_ieeg()
            index = fname.rfind(ieeg_format)
            if index == -1:
                fname += ieeg_format
            raw.save(fname, verbose='error', overwrite=True)
            logger.info('Exporting iEEG to FIF finished!')
            QMessageBox.information(self, 'SEEG', 'Exporting iEEG to FIF finished!')
        else:
            logger.info('Stop exporting iEEG')

    def _export_ieeg_edf(self):
        ieeg_format = '.edf'
        default_fname = os.path.join(default_path, self.subject.get_name() + ieeg_format)
        fname, _ = QFileDialog.getSaveFileName(self, 'iEEG', default_fname, filter=f"EDF+ (*{ieeg_format})")
        if len(fname):
            raw = self.subject.get_ieeg()
            index = fname.rfind(ieeg_format)
            if index == -1:
                fname += ieeg_format
            raw.export(fname, verbose='error', overwrite=True)
            logger.info('Exporting iEEG to EDF finished!')
            QMessageBox.information(self, 'SEEG', 'Exporting iEEG to EDF finished!')
        else:
            logger.info('Stop exporting iEEG')

    def _export_ieeg_set(self):
        ieeg_format = '.set'
        default_fname = os.path.join(default_path, self.subject.get_name() + ieeg_format)
        fname, _ = QFileDialog.getSaveFileName(self, 'iEEG', default_fname, filter=f"EEGLAB (*{ieeg_format})")
        if len(fname):
            raw = self.subject.get_ieeg()
            index = fname.rfind(ieeg_format)
            if index == -1:
                fname += ieeg_format
            raw.export(fname, verbose='error', overwrite=True)
            logger.info('Exporting iEEG to SET finished!')
            QMessageBox.information(self, 'SEEG', 'Exporting iEEG to SET finished!')
        else:
            logger.info('Stop exporting iEEG')

    def _export_coords(self):
        ch_info = self.electrodes.get_info()
        if ch_info is not None:
            fname, _ = QFileDialog.getSaveFileName(self, 'Coordinates', default_path, filter="Coordinates (*.txt)")
            if len(fname):
                if 'txt' not in fname:
                    fname += '.txt'
                ch_info.to_csv(fname, index=None, sep='\t')

    def _clear_mri(self):
        self.subject.remove_t1()
        self.mri_title = ''
        self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                            f'{self.ct_title}')
        logger.info('clear T1 MRI')

    def _clear_ct(self):
        self.subject.remove_ct()
        self.ct_title = ''
        self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                            f'{self.ct_title}')
        logger.info('clear CT')

    def _clear_coordinates(self):
        self.subject.remove_electrodes()

        self.electrodes = Electrodes()
        self.wm_chs = list()
        self.gm_chs = list()
        self.unknown_chs = list()
        self.set_montage = False
        self.parcellation = 'aparc+aseg.vep'

        gc.collect()

        logger.info('clear Coordinates')

    def _clear_ieeg(self):
        self.subject.remove_ieeg()
        try:
            widget = self._ieeg_viz_stack.widget(0)
            self._ieeg_viz_stack.removeWidget(widget)
            self.ieeg_title = ''
            self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                                f'{self.ct_title}')
            logger.info('clear iEEG')
        except:
            logger.info('What??????')

    def _write_info(self):
        self.wins['info_win'] = InfoWin(self._info)
        self.wins['info_win'].INFO_PARAM.connect(self.get_info)
        self.wins['info_win'].show()

    def get_info(self, info):
        self._info = info
        if len(info['subject_name']):
            self.subject.set_name(info['subject_name'])
            print(f"Update subject's info to {info}")

    # View Menu
    def _view_ieeg_info(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            info = dict()
            if isinstance(ieeg, BaseRaw):
                info['epoch_num'] = 1
            info['ch_num'] = len(ieeg.ch_names)
            try:
                info['ch_group'] = len(get_chan_group(ieeg.ch_names))
            except:
                info['ch_group'] = len(ieeg.ch_names)
            info['time'] = round(ieeg.times[-1])
            info['sfreq'] = int(ieeg.info['sfreq'])
            info['fmin'] = ieeg.info['highpass']
            info['fmax'] = int(ieeg.info['lowpass'])
            info['data_size'] = round(ieeg._size / (1024 ** 2), 2)
            self.wins['ieeg_info_win'] = iEEGInfoWin(info)
            self.wins['ieeg_info_win'].show()
        else:
            QMessageBox.warning(self, 'iEEG', 'Please load iEEG first!')

    def _view_chs_info(self):
        ch_info = self.electrodes.get_info()
        if len(ch_info):
            self.wins['table_win'] = TableWin(ch_info, self.chs_name)
            self.wins['table_win'].show()

    # Localization Menu
    def _display_t1(self):
        logger.info('Display T1 MRI')
        plt.style.use('dark_background')
        t1 = self.subject.get_t1()
        if t1 is not None:
            self.fig, self.t1_axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
            self.t1_viewer = OrthoSlicer3D(t1.dataobj, t1.affine,
                                           axes=self.t1_axes, title='T1 MRI')
            self.t1_axes[0].set_title('Sagittal', pad=20)
            self.t1_axes[1].set_title('Coronal', pad=20)
            self.t1_axes[2].set_title('Axial', pad=20)
            self.t1_viewer.show()

    def _display_ct(self):
        logger.info('Display CT')
        plt.style.use('dark_background')
        ct = self.subject.get_ct()
        if ct is not None:
            self.fig, self.ct_axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
            self.ct_viewer = OrthoSlicer3D(ct.dataobj, ct.affine,
                                           axes=self.ct_axes, title='CT')
            self.ct_axes[0].set_title('Sagittal', pad=20)
            self.ct_axes[1].set_title('Coronal', pad=20)
            self.ct_axes[2].set_title('Axial', pad=20)
            self.ct_viewer.show()

    def _plot_overlay(self):
        logger.info('Display Overlay')
        plt.style.use('dark_background')

        thresh = 0.95

        t1 = self.subject.get_t1()
        ct = self.subject.get_align_ct()
        if ct is None:
            ct = self.subject.get_ct()

        if t1 is not None and ct is not None:
            if np.asarray(t1.dataobj).shape != np.asarray(ct.dataobj).shape:
                ct = resample(moving=np.asarray(ct.dataobj),
                              static=np.asarray(t1.dataobj),
                              moving_affine=ct.affine,
                              static_affine=t1.affine)

            t1 = nib.orientations.apply_orientation(
                np.asarray(t1.dataobj), nib.orientations.axcodes2ornt(
                    nib.orientations.aff2axcodes(t1.affine))).astype(np.float32)

            ct = nib.orientations.apply_orientation(
                np.asarray(ct.dataobj), nib.orientations.axcodes2ornt(
                    nib.orientations.aff2axcodes(ct.affine))).astype(np.float32)
            if thresh is not None:
                ct[ct < np.quantile(ct, thresh)] = np.nan

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle('aligned CT Overlaid on T1')
            for i, ax in enumerate(axes):
                ax.imshow(np.take(t1, [t1.shape[i] // 2], axis=i).squeeze().T,
                          cmap='gray')
                ax.imshow(np.take(ct, [ct.shape[i] // 2],
                                  axis=i).squeeze().T, cmap='hsv', alpha=0.5)
                ax.invert_yaxis()
                ax.axis('off')
            fig.tight_layout()
            plt.show()
            print('Finish displaying overlay')

    def _locate_ieeg(self):
        raw = self.subject.get_ieeg()
        name = self.subject.get_name()
        if raw is None:
            QMessageBox.warning(self, 'iEEG', 'Please import an iEEG data first!')
            return
        if name is None:
            QMessageBox.warning(self, 'Subject', 'Please set subject name first!')
            return
        subj_trans = mne.coreg.estimate_head_mri_t(name, self.subjects_dir)

        t1 = self.subject.get_t1()
        if t1 is None:
            QMessageBox.warning(self, 'MRI', 'Please import MRI first!')
            return

        ct = self.subject.get_align_ct()
        if ct is None:
            ct = self.subject.get_ct()
        if ct is None:
            QMessageBox.warning(self, 'CT', 'Please import CT first!')
            return

        if np.asarray(t1.dataobj).shape != np.asarray(ct.dataobj).shape:
            ct = resample(moving=np.asarray(ct.dataobj),
                          static=np.asarray(t1.dataobj),
                          moving_affine=ct.affine,
                          static_affine=t1.affine)

        self.ieeg_locator = locate_ieeg(raw.info, subj_trans, ct,
                                        subject=name, subjects_dir=self.subjects_dir)
        self.ieeg_locator.CLOSE_SIGNAL.connect(self.get_contact_pos)
        logger.info('Start iEEG Locator')

    def get_contact_pos(self, result):
        ch_names = []
        x = []
        y = []
        z = []
        ch_pos = {}
        for ch in result:
            xyz = np.round(result[ch], 3)
            if True not in np.isnan(xyz):
                ch_pos[ch] = xyz
                ch_names.append(ch)

        raw = self.subject.get_ieeg()
        subject = self.subject.get_name()
        ch_locate_group = get_chan_group(ch_names)
        ch_group = get_chan_group(raw.ch_names)
        contact_pos = ch_pos.copy()
        for group in ch_group:
            tip_ch = ch_group[group][0]
            tail_ch = ch_group[group][-1]
            if (tip_ch in ch_pos.keys()) and \
                    (tail_ch in ch_pos.keys()) and (len(ch_locate_group[group]) == 2):
                tip = ch_pos[tip_ch]
                tail = ch_pos[tail_ch]
                ch_num = len(ch_group[group])
                dist = 3.5
                print(f'Calculating group {group} contacts based on {tip_ch} and {tail_ch}')
                calc_pos = calc_ch_pos(tip, tail, ch_num, dist)
                curr_ch_names = ch_group[group]
                contact_pos.update(dict(zip(curr_ch_names, calc_pos)))

        if len(contact_pos):
            montage_pos = {ch: contact_pos[ch] / 1000 for ch in contact_pos}
            _, montage = get_montage(montage_pos, subject, self.subjects_dir)
            raw.set_montage(montage, on_missing='ignore')

            self.electrodes.clean()
            ch_names = list(contact_pos.keys())
            ch_names = reorder_chs(ch_names)
            self.electrodes.set_ch_names(ch_names)
            for ch in ch_names:
                xyz = np.round(contact_pos[ch], 3)
                x.append(xyz[0])
                y.append(xyz[1])
                z.append(xyz[2])
            self.electrodes.set_ch_xyz([x, y, z])

    # Signal Menu
    def _set_ieeg_montage(self):
        ch_info = self.electrodes.get_info()
        ieeg = self.subject.get_ieeg()
        subject = self.subject.get_name()
        if not len(ch_info):
            QMessageBox.warning(self, 'Coordinates', 'Please load Coordinates first!')
            return
        if ieeg is None:
            QMessageBox.warning(self, 'iEEG', 'Please load iEEG first!')
            return

        ch_names = ch_info['Channel'].to_list()

        if subject is not None:
            xyz = ch_info[['x', 'y', 'z']].to_numpy() / 1000.
            ch_pos = dict(zip(ch_names, xyz))
            ieeg = set_montage(ieeg, ch_pos, subject, self.subjects_dir)
            self.subject.set_ieeg(ieeg)
        self.subject.set_electrodes(self.electrodes.get_info())

        # This happens when loading coordinates with anatomy or
        # getting the anatomy before setting the montage
        if 'issue' in ch_info.columns:
            issue = self.electrodes.get_issue()
            ch_names = np.array(ch_names)
            self.wm_chs = ch_names[issue == 'White']
            self.unknown_chs = ch_names[issue == 'Unknown']
            self.gm_chs = ch_names[issue == 'Gray']

        self.set_montage = True

        logger.info('Set iEEG montage finished!')
        QMessageBox.information(self, 'Montage', 'Set iEEG montage finished!')

    def _crop_ieeg(self):
        if self.subject.get_ieeg() is not None:
            raw = self.subject.get_ieeg()
            self.wins['crop_win'] = CropWin(raw.times[0], round(raw.times[-1]))
            self.wins['crop_win'].CROP_SIGNAL.connect(self._get_cropped_ieeg)
            self.wins['crop_win'].show()

    def _get_cropped_ieeg(self, tmin, tmax):
        self.subject.get_ieeg().crop(tmin, tmax)
        QMessageBox.information(self, 'iEEG', 'Finish Cropping iEEG!')
        self._update_fig()

    def _resample_ieeg(self):
        print('here1')
        if self.subject.get_ieeg() is not None:
            print('here2')
            raw = self.subject.get_ieeg()
            self.wins['resample_win'] = ResampleWin(raw)
            self.wins['resample_win'].RESAMPLE_SIGNAL.connect(self._get_resampled_ieeg)
            self.wins['resample_win'].show()

    def _get_resampled_ieeg(self, ieeg):
        QMessageBox.information(self, 'iEEG', 'Resampling finished!')
        self.subject.set_ieeg(ieeg)
        self._update_fig()

    def _fir_filter_ieeg(self):
        raw = self.subject.get_ieeg()
        if raw is not None:
            self.wins['fir_filter_win'] = FIRFilterWin(raw)
            self.wins['fir_filter_win'].IEEG_SIGNAL.connect(self._get_filtered_ieeg)
            self.wins['fir_filter_win'].show()

    def _get_filtered_ieeg(self, ieeg):
        QMessageBox.information(self, 'iEEG', 'Filtering finished!')
        self.subject.set_ieeg(ieeg)
        self._update_fig()

    def _monopolar_reference(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['monopolar_win'] = ItemSelectionWin(ieeg.ch_names)
            self.wins['monopolar_win'].SELECTION_SIGNAL.connect(self._get_monopolar_chans)
            self.wins['monopolar_win'].show()

    def _get_monopolar_chans(self, chans):
        ieeg, _ = mne.set_eeg_reference(self.subject.get_ieeg(), ref_channels=chans, copy=False)
        self.subject.set_ieeg(ieeg)
        self._update_fig()
        QMessageBox.warning(self, 'iEEG', f'Reference iEEG based on {chans} finished!')

    def _bipolar_ieeg(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            ieeg = mne_bipolar(ieeg)
            self.subject.set_ieeg(ieeg)
            ch_info = self.subject.get_electrodes()
            if ch_info is not None:
                print('ch_info is not None')

                ch_names = ch_info['Channel'].to_list()
                xyz = ch_info[['x', 'y', 'z']].to_numpy()
                ch_pos = dict(zip(ch_names, xyz))
                bipolar_ch_pos = calc_bipolar_chs_pos(ch_pos)

                ch_names = list(bipolar_ch_pos.keys())

                bipolar_xyz = np.asarray(list(bipolar_ch_pos.values()))
                x = bipolar_xyz[:, 0]
                y = bipolar_xyz[:, 1]
                z = bipolar_xyz[:, 2]

                bipolar_ch_df = pd.DataFrame()
                bipolar_ch_df['Channel'] = ch_names
                bipolar_ch_df['x'] = x
                bipolar_ch_df['y'] = y
                bipolar_ch_df['z'] = z
                self.subject.set_electrodes(bipolar_ch_df)
                self.electrodes.clean()
                self.electrodes.set_ch_names(ch_names)
                self.electrodes.set_ch_xyz([x, y, z])
                logger.info("Finish bipolar channels coordinates")
            else:
                self.subject.set_electrodes(None)
            self._update_fig()

    def _laplacian_ieeg(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            ieeg = set_laplacian_ref(ieeg)
            self.subject.set_ieeg(ieeg)
            self._update_fig()

    def _average_reference(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            ieeg, _ = mne.set_eeg_reference(ieeg, ref_channels='average', copy=False)
            self.subject.set_ieeg(ieeg)
            self._update_fig()
            QMessageBox.warning(self, 'iEEG', f'Common average reference iEEG finished!')

    def _drop_bad_from_annotations(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            fig = self._ieeg_viz_stack.widget(0)
            fig.close() # You'll have to close the fig to get the bad channels
            bad = ieeg.info['bads']
            if len(bad):
                try:
                    self.subject.get_ieeg().drop_channels(bad)
                    self.subject.get_ieeg().info['bads'] = []
                    logger.info(f'Dropping bad channels: {bad} finished!')
                    self._update_fig()
                    QMessageBox.information(self, 'iEEG', f'Finish dropping bad channels {bad}')
                except:
                    logger.warning('No channels will be left, so dropping channels is stopped')
            else:
                logger.info('No bad channels in annotations!')
                self._update_fig()
                QMessageBox.information(self, 'iEEG', 'No bad channels in annotations')

    def _drop_wm_chs(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is None:
            return
        if len(self.wm_chs):
            logger.info(f'Drop white matter channels: {self.wm_chs}')
            # try:
            drop_chans = list(set(self.wm_chs).intersection(set(ieeg.ch_names)))
            if len(drop_chans):
                ieeg.drop_channels(drop_chans)
                self._update_fig()
            # except:
            #     logger.warning('No channels will be left, so dropping channels is stopped')

    def _drop_gm_chs(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is None:
            return
        if len(self.gm_chs):
            logger.info(f'Drop gray matter channels: {self.gm_chs}')
            # try:
            drop_chans = list(set(self.gm_chs).intersection(set(ieeg.ch_names)))
            if len(drop_chans):
                ieeg.drop_channels(drop_chans)
                self._update_fig()
            # except:
            #     logger.warning('No channels will be left, so dropping channels is stopped')

    def _drop_unknown_chs(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is None:
            return
        if len(self.unknown_chs):
            logger.info(f'Drop unknown channels: {self.unknown_chs}')
            # try:
            drop_chans = list(set(self.unknown_chs).intersection(set(ieeg.ch_names)))
            if len(drop_chans):
                ieeg.drop_channels(drop_chans)
                self._update_fig()
            # except:
            #     logger.warning('No channels will be left, so dropping channels is stopped')

    # Analysis Menu
    def _set_anatomy(self):
        mri_path, _ = QFileDialog.getOpenFileName(self, 'Anatomy', freesurfer_path,
                                                  filter='Anatomy (*.mgz *.nii)')
        self.mri_path = mri_path
        self._get_anatomy()

    def _get_anatomy(self):
        ch_info = self.electrodes.get_info()

        if not len(ch_info):
            QMessageBox.warning(self, 'Coordinates', 'Please Load Coordinates first!')
            return

        if 'x' not in ch_info.columns:
            QMessageBox.warning(self, 'Coordinates', 'Please Load Coordinates first!')
            return
        xyz = ch_info[['x', 'y', 'z']].to_numpy()
        ch_names = ch_info['Channel'].to_numpy()

        if 'vep' in self.mri_path:
            lut_path = 'utils/VepFreeSurferColorLut.txt'
        else:
            lut_path = 'utils/FreeSurferColorLUT.txt'

        if op.isfile(self.mri_path):
            anatomical_labels = labelling_contacts_vol_fs_mgz(self.mri_path, xyz, radius=2, lut_path=lut_path)
            self.electrodes.set_issues(anatomical_labels)
            self.electrodes.set_anatomy('ROI', anatomical_labels)

            # if set_montage is True, replace the electrodes info in subject
            # This happens when loading coordinates without anatomy and setting the montage
            # before getting the anatomy
            if self.set_montage:
                self.subject.set_electrodes(self.electrodes.get_info())
                issue = self.electrodes.get_issue()
                self.wm_chs = ch_names[issue == 'White']
                self.unknown_chs = ch_names[issue == 'Unknown']
                self.gm_chs = ch_names[issue == 'Gray']
        else:
            QMessageBox.critical(self, 'Anatomy', f'{self.mri_path} not exists')

    def _psd_multitaper(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['psd_multitaper_win'] = MultitaperPSDWin(ieeg)
            self.wins['psd_multitaper_win'].show()

    def _psd_welch(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self._psd_welch_win = WelchPSDWin(ieeg)
            self._psd_welch_win.show()

    def _csd_fourier(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['csd_fourier_win'] = FourierCSDWin(ieeg)
            self.wins['csd_fourier_win'].show()

    def _csd_morlet(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['csd_fourier_win'] = MorletCSDWin(ieeg)
            self.wins['csd_fourier_win'].show()

    def _csd_multitaper(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['csd_fourier_win'] = MultitaperCSDWin(ieeg)
            self.wins['csd_fourier_win'].show()

    def _tfr_multitaper(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['tfr_multitaper_win'] = TFRMultitaperWin(ieeg)
            self.wins['tfr_multitaper_win'].show()

    def _tfr_morlet(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['tfr_morlet_win'] = TFRMorletWin(ieeg)
            self.wins['tfr_morlet_win'].show()

    def _nxn_con(self, method):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['nxn_con_win'] = NxNSpectraConWin(ieeg, method)
            self.wins['nxn_con_win'].show()

    def _nx1_con(self, method):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self.wins['nx1_con_win'] = Nx1SpectraConWin(ieeg, method)
            self.wins['nx1_con_win'].show()

    def _transfer_anatomy(self, win_type):
        windows = {
            'EI': self.wins['ei_win'],
            'HFO': self.wins['hfo_win'],
        }
        win = windows[win_type]
        ch_info = self.subject.get_electrodes()
        subject = self.subject.get_name()
        if ch_info is not None:
            print('Transfer anatomy to sub window')
            if 'issues' in ch_info.columns:
                win.seg_name = self.seg_name[self.parcellation]
                win.parcellation = self.parcellation
                win.set_anatomy(subject, self.subjects_dir, ch_info)

    def _compute_ei(self):
        subject = self.subject.get_name()
        ieeg = self.subject.get_ieeg()
        ch_info = self.subject.get_electrodes()
        anatomy = None
        if ch_info is not None:
            if 'issue' in ch_info.columns:
                anatomy = ch_info[['Channel', 'x', 'y', 'z', self.seg_name[self.parcellation]]]
                seg_name = self.seg_name[self.parcellation]
                # anatomy = ch_info[['Channel', 'x', 'y', 'z', seg_name]]
        if ieeg is not None:
            self.wins['ei_win'] = EIWin(ieeg, subject, self.subjects_dir, anatomy, seg_name, self.parcellation)
            self.wins['ei_win'].ANATOMY_SIGNAL.connect(self._transfer_anatomy)
            self.wins['ei_win'].show()

    def _compute_hfo(self):
        ieeg = self.subject.get_ieeg()
        ch_info = self.subject.get_electrodes()
        anatomy = None
        if ch_info is not None:
            if 'issue' in ch_info.columns:
                anatomy = ch_info[['Channel', 'x', 'y', 'z', self.seg_name[self.parcellation]]]
                seg_name = self.seg_name[self.parcellation]
                # anatomy = ch_info[['Channel', 'x', 'y', 'z', seg_name]]
        if ieeg is not None:
            self.wins['hfo_win'] = RMSHFOWin(ieeg, anatomy, seg_name)
            self.wins['hfo_win'].ANATOMY_SIGNAL.connect(self._transfer_anatomy)
            self.wins['hfo_win'].show()

    # Visualization Menu
    def _electrodes_viz(self):
        subject = self.subject.get_name()
        ch_info = self.electrodes.get_info()
        columns = list(ch_info.columns)
        remove_columns = ['Channel', 'x', 'y', 'z']
        for rm in remove_columns:
            columns.remove(rm)
        values = self.seg_name.values()
        seg_name = None
        for value in values:
            if value in columns:
                seg_name = value

                key_list = list(self.seg_name.keys())
                value_list = list(self.seg_name.values())

                position = value_list.index(seg_name)
                self.parcellation = key_list[position]

        if subject is not None:
            self.wins['elec_viz_win'] = ElectrodesWin(subject, freesurfer_path, ch_info, seg_name,
                                               self.parcellation)
            self.wins['elec_viz_win'].CLOSE_SIGNAL.connect(self._clean_elec_viz_win)
            self.wins['elec_viz_win'].show()

    def _clean_elec_viz_win(self, close):
        if close:
            self.wins['elec_viz_win'] = None

    def _rois_viz(self):
        subject = self.subject.get_name()
        ch_info = self.electrodes.get_info()
        columns = list(ch_info.columns)
        remove_columns = ['Channel', 'x', 'y', 'z']
        for rm in remove_columns:
            columns.remove(rm)
        values = self.seg_name.values()
        seg_name = None
        for value in values:
            if value in columns:
                seg_name = value
        if seg_name is None:
            QMessageBox.warning(self, 'ROIs', 'Please get anatomy of electrodes first!')
            return

        key_list = list(self.seg_name.keys())
        value_list = list(self.seg_name.values())

        position = value_list.index(seg_name)
        self.parcellation = key_list[position]

        ch_info = ch_info.rename(columns={seg_name: 'ROI'})
        if subject is not None:
            self.wins['rois_viz_win'] = ROIsWin(subject, freesurfer_path, ch_info, self.parcellation)
            self.wins['rois_viz_win'].CLOSE_SIGNAL.connect(self._clean_roi_viz_win)
            self.wins['rois_viz_win'].show()

    def _clean_roi_viz_win(self, close):
        if close:
            self.wins['rois_viz_win'] = None

    def _open_freeview(self):
        if SYSTEM != 'Windows':
            result = os.system('freeview')
            if result:
                QMessageBox.warning(self, 'FreeSurfer', 'Please install Freesurfer!')

    # Toolbar
    def _take_screenshot(self):
        if self._ieeg_viz_stack.count() > 0:
            fig = self._ieeg_viz_stack.widget(0)
            fpath, _ = QFileDialog.getSaveFileName(self, 'Screenshot', default_path)
            if len(fpath):
                fig.mne.view.grab().save(f'{fpath}.JPEG')
                logger.info(f'take a screenshot to {fpath}')

    def _viz_ieeg_toolbar(self):
        if self._ieeg_viz_stack.count() > 0:
            fig = self._ieeg_viz_stack.widget(0)
            viz = fig.mne.toolbar.isVisible()
            fig.mne.toolbar.setVisible(not viz)

    # Help Menu
    @staticmethod
    def _open_github():
        logger.info('Open github website!')
        url = QUrl('https://github.com/BarryLiu97/iEEGTool')
        QDesktopServices.openUrl(url)

    @safe_event
    def closeEvent(self, event):
        [self.wins[win].close() for win in self.wins if self.wins[win] is not None]