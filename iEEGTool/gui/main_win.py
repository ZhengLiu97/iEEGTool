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
from gui.info_win import InfoWin
from gui.list_win import ItemSelectionWin
from gui.fir_filter_win import FIRFilterWin
from gui.compute_ei_win import EIWin
# from gui.compute_hfo_win import RMSHFOWin
from gui.table_win import TableWin
from gui.tfr_morlet_win import TFRMorletWin
from utils.subject import Subject
from utils.thread import *
from utils.log_config import create_logger
from utils.decorator import safe_event
from utils.locate_ieeg import locate_ieeg
from utils.contacts import calc_ch_pos
from utils.process import get_chan_group, set_montage, clean_chans, get_montage, mne_bipolar

matplotlib.use('Qt5Agg')
mne.viz.set_browser_backend('pyqtgraph')

SYSTEM = platform.system()

logger = create_logger(filename='iEEGTool.log')

default_path = 'data'


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
        self.subject = Subject('')
        self._info = {'subject_name': '', 'age': '', 'gender': ''}
        self.subjects_dir = op.join(default_path, 'freesurfer')

        self._crop_win = None
        self._resample_win = None
        self._fir_filter_win = None
        self._iir_filter_win = None
        self._tfr_morlet_win = None

        self._ei_win = None
        self._hfo_win = None

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

        self._clear_mri_action.triggered.connect(self._clear_mri)
        self._clear_ct_action.triggered.connect(self._clear_ct)
        self._clear_coordinate_action.triggered.connect(self._clear_coordinates)
        self._clear_ieeg_action.triggered.connect(self._clear_ieeg)

        self._setting_action.triggered.connect(self._write_info)

        # View Menu
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
        self._average_action.triggered.connect(self._average_reference)
        self._drop_annotations_action.triggered.connect(self._drop_bad_from_annotations)

        # Analysis Menu
        self._get_anatomy_action.triggered.connect(self._get_anatomy)
        self._tfr_morlet_action.triggered.connect(self._tfr_morlet)
        self._epileptogenic_index_action.triggered.connect(self._compute_ei)
        self._high_frequency_action.triggered.connect(self._compute_hfo)

        # Help Menu
        self._github_action.triggered.connect(self._open_github)

        # Toolbar
        self._screenshot_action.triggered.connect(self._take_screenshot)
        self._ieeg_toolbar_action.triggered.connect(self._viz_ieeg_toolbar)

    def _update_fig(self):
        raw = self.subject.get_ieeg()
        fig = mne.viz.plot_raw(raw, remove_dc=True, color='k',
                               n_channels=30, scalings='auto', show=False)
        fig.mne.overview_bar.setVisible(False)
        fig.statusBar().setVisible(False)
        fig._set_annotations_visible(False)
        fig.mne.toolbar.setVisible(False)

        if self._ieeg_viz_stack.count() > 0:
            widget = self._ieeg_viz_stack.widget(0)
            self._ieeg_viz_stack.removeWidget(widget)
        self._ieeg_viz_stack.addWidget(fig)

    def _short_cut(self):
        # QShortcut(QKeySequence(self.tr("Ctrl+O")), self, self._import_ieeg)
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
        logger.info(f'Cleaning channels finished!')
        self.subject.set_ieeg(ieeg)
        self._update_fig()
        self.setWindowTitle(f'iEEG Tool      {self.ieeg_title}   {self.mri_title}   '
                            f'{self.ct_title}')
        logger.info('Set iEEG')

    def _import_coord(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Coordinates', default_path,
                                               filter='Coordinates (*.txt *.tsv)')
        if len(fname):
            try:
                coords_df = pd.read_table(fname)
                ch_names = coords_df['Channel'].to_list()
                x = coords_df['x'].to_numpy()
                y = coords_df['y'].to_numpy()
                z = coords_df['z'].to_numpy()
                ch_pos = pd.DataFrame()
                ch_pos['Channel'] = ch_names
                ch_df = get_chan_group(chans=ch_names, return_df=True)
                group = ch_df['Group'].to_list()
                ch_pos['Group'] = group
                ch_pos['x'] = x
                ch_pos['y'] = y
                ch_pos['z'] = z
                self.subject.set_electrodes(ch_pos)
                logger.info("Importing channels' coordinates finished!")
                QMessageBox.information(self, 'Coordinates', "Importing channels' coordinates finished!")
            except:
                QMessageBox.warning(self, 'Coordinates', 'Wrong file format!')

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
        self.subject.remove_anatomy()
        self.subject.remove_anatomy_electrodes()
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
        self._info_win = InfoWin(self._info)
        self._info_win.INFO_PARAM.connect(self.get_info)
        self._info_win.show()

    def get_info(self, info):
        self._info = info
        self.subject.set_name(info['subject_name'])
        logger.info(f"Update subject's info to {info}")

    # View Menu
    def _view_chs_info(self):
        ch_pos = self.subject.get_electrodes()
        if ch_pos is not None:
            self._table_win = TableWin(ch_pos)
            self._table_win.show()

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

    def get_contact_pos(self, event):
        raw = self.subject.get_ieeg()
        subject = self.subject.get_name()
        ch_names, chs = raw.info['ch_names'], raw.info['chs']
        pos = np.asarray([chs[i]['loc'][:3] for i in range(len(ch_names))])
        ch_pos_df = pd.DataFrame()
        ch_pos_df['Channel'] = ch_names
        ch_pos_df['x'] = pos[:, 0]
        ch_pos_df['y'] = pos[:, 1]
        ch_pos_df['z'] = pos[:, 2]
        ch_pos_df = ch_pos_df.dropna(axis=0, how='any')

        ch_locate = ch_pos_df['Channel'].to_list()
        pos = ch_pos_df[['x', 'y', 'z']].to_numpy()
        ch_pos = dict(zip(ch_locate, pos))
        ch_locate_group = get_chan_group(ch_locate)
        ch_group = get_chan_group(raw.ch_names)
        contact_pos = ch_pos.copy()
        for group in ch_group:
            tip_ch = ch_group[group][0]
            tail_ch = ch_group[group][-1]
            if (tip_ch in ch_pos.keys()) and \
                    (tail_ch in ch_pos.keys()) and (len(ch_locate_group[group]) == 2):
                tip = ch_pos[tip_ch] * 1000
                tail = ch_pos[tail_ch] * 1000
                ch_num = len(ch_group[group])
                dist = 3.5
                print(f'Calculating group {group} contacts')
                calc_pos = calc_ch_pos(tip, tail, ch_num, dist)
                curr_ch_names = ch_group[group]
                contact_pos.update(dict(zip(curr_ch_names, calc_pos / 1000)))

        if len(contact_pos):
            lpa, nasion, rpa = mne.coreg.get_mni_fiducials(
                subject=subject, subjects_dir=self.subjects_dir)
            lpa, nasion, rpa = lpa['r'], nasion['r'], rpa['r']
            montage = mne.channels.make_dig_montage(
                contact_pos, coord_frame='head', nasion=nasion, lpa=lpa, rpa=rpa)
            raw.set_montage(montage, on_missing='ignore')

    # Signal Menu
    def _set_ieeg_montage(self):
        ch_pos = self.subject.get_electrodes()
        ieeg = self.subject.get_ieeg()
        subject = self.subject.get_name()
        if ch_pos is None:
            QMessageBox.warning(self, 'Coordinates', 'Please load Coordinates first!')
            return
        if ieeg is None:
            QMessageBox.warning(self, 'iEEG', 'Please load iEEG first!')
            return
        if subject is None:
            QMessageBox.warning(self, 'Subject', "Please set Subject's name first!")
            return
        ch_names = ch_pos['Channel'].to_list()
        xyz = ch_pos[['x', 'y', 'z']].to_numpy() / 1000.
        ch_pos = dict(zip(ch_names, xyz))

        # subj_trans = mne.coreg.estimate_head_mri_t(subject, self.subjects_dir)
        # mri_to_head_trans = mne.transforms.invert_transform(subj_trans)
        # print('Start transforming mri to head')
        # print(mri_to_head_trans)
        #
        # montage = mne.channels.make_dig_montage(ch_pos, coord_frame='mri')
        # montage.add_estimated_fiducials(subject, subjects_dir)
        # montage.apply_trans(mri_to_head_trans)
        # self.subject.get_ieeg()._montage(montage, on_missing='ignore')

        ieeg = set_montage(ieeg, ch_pos, subject, self.subjects_dir)
        self.subject.set_ieeg(ieeg)
        logger.info('Set iEEG montage finished!')
        QMessageBox.information(self, 'Montage', 'Set iEEG montage finished!')

    def _crop_ieeg(self):
        if self.subject.get_ieeg() is not None:
            raw = self.subject.get_ieeg()
            self._crop_win = CropWin(raw.times[0], round(raw.times[-1]))
            self._crop_win.CROP_SIGNAL.connect(self._get_cropped_ieeg)
            self._crop_win.show()

    def _get_cropped_ieeg(self, tmin, tmax):
        self.subject.get_ieeg().crop(tmin, tmax)
        QMessageBox.information(self, 'iEEG', 'Finish Cropping iEEG!')
        self._update_fig()

    def _resample_ieeg(self):
        if self.subject.get_ieeg() is not None:
            raw = self.subject.get_ieeg()
            self._resample_win = ResampleWin(raw)
            self._resample_win.RESAMPLE_SIGNAL.connect(self._get_resampled_ieeg)
            self._resample_win.show()

    def _get_resampled_ieeg(self, ieeg):
        QMessageBox.information(self, 'iEEG', 'Resampling finished!')
        self.subject.set_ieeg(ieeg)
        self._update_fig()

    def _fir_filter_ieeg(self):
        raw = self.subject.get_ieeg()
        if raw is not None:
            self._fir_filter_win = FIRFilterWin(raw)
            self._fir_filter_win.IEEG_SIGNAL.connect(self._get_filtered_ieeg)
            self._fir_filter_win.show()

    def _get_filtered_ieeg(self, ieeg):
        QMessageBox.information(self, 'iEEG', 'Filtering finished!')
        self.subject.set_ieeg(ieeg)
        self._update_fig()

    def _monopolar_reference(self):
        self._monopolar_win = ItemSelectionWin(self.subject.get_ieeg().ch_names)
        self._monopolar_win.SELECTION_SIGNAL.connect(self._get_monopolar_chans)
        self._monopolar_win.show()

    def _get_monopolar_chans(self, chans):
        ieeg, _ = mne.set_eeg_reference(self.subject.get_ieeg(), ref_channels=chans, copy=False)
        self.subject.set_ieeg(ieeg)
        self._update_fig()
        QMessageBox.warning(self, 'iEEG', f'Reference iEEG based on {chans} finished!')

    def _bipolar_ieeg(self):
        ieeg = mne_bipolar(self.subject.get_ieeg())
        self.subject.set_ieeg(ieeg)
        self._update_fig()

    def _average_reference(self):
        ieeg, _ = mne.set_eeg_reference(self.subject.get_ieeg(), ref_channels='average', copy=False)
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
                self.subject.get_ieeg().drop_channels(bad)
                self.subject.get_ieeg().info['bads'] = []
                logger.info(f'Dropping bad channels: {bad} finished!')
                self._update_fig()
                QMessageBox.information(self, 'iEEG', f'Finish dropping bad channels {bad}')
            else:
                logger.info('No bad channels in annotations!')
                self._update_fig()
                QMessageBox.information(self, 'iEEG', 'No bad channels in annotations')

    # Analysis Menu
    def _get_anatomy(self):
        ch_pos_df = self.subject.get_electrodes()
        if ch_pos_df is None:
            QMessageBox.warning(self, 'Coordinates', 'Please Load Coordinates first!')
            return

    def _tfr_morlet(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self._tfr_morlet_win = TFRMorletWin(ieeg)
            self._tfr_morlet_win.show()

    def _compute_ei(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self._ei_win = EIWin(ieeg)
            self._ei_win.show()

    def _compute_hfo(self):
        ieeg = self.subject.get_ieeg()
        if ieeg is not None:
            self._hfo_win = RMSHFOWin(ieeg)
            self._hfo_win.show()

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
        if self._crop_win is not None:
            self._crop_win.close()
        if self._resample_win is not None:
            self._resample_win.close()
        if self._fir_filter_win is not None:
            self._fir_filter_win.close()
        if self._ei_win is not None:
            self._ei_win.close()
        if self._hfo_win is not None:
            self._hfo_win.close()
        if self._tfr_morlet_win is not None:
            self._tfr_morlet_win.close()