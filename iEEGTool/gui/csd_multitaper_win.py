# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：csd_multitaper_win.py
@Author  ：Barry
@Date    ：2022/3/11 18:51 
"""
import mne
import numpy as np

from mne.io import BaseRaw
from mne.time_frequency.multitaper import _compute_mt_params
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QButtonGroup, QMessageBox

from gui.csd_multitaper_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from utils.process import make_epoch
from utils.log_config import create_logger
from utils.thread import ComputeCSD
from viz.figure import create_heatmap

logger = create_logger(filename='iEEGTool.log')


class MultitaperCSDWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Cross Spectral Density')

        if isinstance(ieeg, BaseRaw):
            ieeg = make_epoch(ieeg)
        self.ieeg = ieeg

        fmin = str(int(ieeg.info['highpass']))
        fmax = str(int(ieeg.info['lowpass']))
        fmin = '0.1' if fmin == '0' else fmin
        self._freq_band_le.setText(fmin + ' ' + fmax)

        self._compute_chans = ieeg.ch_names

        self.compute_params = {}
        self.fig_params = {}

        self.csd = None
        self.freqs = None

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _slot_connection(self):
        self._select_chan_btn.clicked.connect(self._select_chan)
        self._compute_btn.clicked.connect(self._compute_csd)
        self._plot_btn.clicked.connect(self._plot_csd)

    def _select_chan(self):
        self._select_chan_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chan_win.SELECTION_SIGNAL.connect(self._get_compute_chans)
        self._select_chan_win.show()

    def _get_compute_chans(self, chans):
        self._compute_chans = chans
        if len(chans) < 2:
            QMessageBox.warning(self, 'CSD', 'As least two channels!')
            return
        logger.info(f"Selected channels are {chans}")

    def get_compute_params(self):
        keep = True
        freq_band = self._freq_band_le.text().split(' ')
        if len(freq_band) != 2:
            QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
            return
        fmin = float(freq_band[0])
        fmax = float(freq_band[1])
        bandwidth = float(self._bandwidth_le.text())
        n_jobs = int(self._n_jobs_le.text())
        self.compute_params['fmin'] = fmin
        self.compute_params['fmax'] = fmax
        self.compute_params['bandwidth'] = bandwidth
        self.compute_params['adaptive'] = True
        self.compute_params['low_bias'] = True
        self.compute_params['n_jobs'] = n_jobs

        n_times = self.ieeg.get_data().shape[-1]
        sfreq = self.ieeg.info['sfreq']
        _, eigvals, _ = _compute_mt_params(n_times, sfreq, bandwidth, True, True)
        question = f'Using multitaper spectrum estimation with {len(eigvals)} DPSS windows'
        logger.info(question)
        reply = QMessageBox.question(self, 'PSD',
                                     question,
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
        if reply == QMessageBox.No:
            keep = False

        return keep

    def get_fig_params(self):
        mean = self._mean_freqs_cbx.currentText()
        self.fig_params['mean'] = True if mean == 'True' else False
        print(self._mean_freq_band_le.text())
        if ' ' in self._mean_freq_band_le.text():
            freq_band = self._mean_freq_band_le.text().split(' ')
            print('here')
            if len(freq_band) != 2:
                QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
                return
        else:
            freq_band = [float(self._mean_freq_band_le.text())]
        self.fig_params['freq_band'] = freq_band

    def _compute_csd(self):
        logger.info('Start computing CSD using Multitaper')
        keep = self.get_compute_params()
        if keep:
            logger.info(f'Compute CSD with \n {self.compute_params}')

            ieeg = self.ieeg.copy()
            if len(self._compute_chans):
                ieeg.pick_channels(self._compute_chans)
            self._compute_csd_multitaper_thread = ComputeCSD(ieeg, compute_method='multitaper',
                                                          params=self.compute_params)
            self._compute_csd_multitaper_thread.CSD_SIGNAL.connect(self._get_csd)
            self._compute_csd_multitaper_thread.start()

    def _get_csd(self, csd):
        self.csd = csd
        self.freqs = csd.frequencies
        self._mean_freq_band_le.setText(self._freq_band_le.text())
        QMessageBox.information(self, 'CSD', 'Finish computing CSD!')

    def _plot_csd(self):
        if self.csd is not None:
            csd = self.csd.copy()
            self.get_fig_params()
            mean = self.fig_params['mean']
            freq_band = self.fig_params['freq_band']
            if len(freq_band) > 2 or len(freq_band) < 1:
                QMessageBox.warning(self, 'CSD', 'Please input a frequency band!')
                return
            elif len(freq_band) == 2:
                freq_band = [float(freq) for freq in freq_band]

            if mean:
                if len(freq_band) == 2:
                    fmin = freq_band[0]
                    fmax = freq_band[1]
                    if fmin > fmax:
                        return
                    csd_data = np.abs(csd.mean(fmin, fmax).get_data())
                    mask = np.zeros_like(csd_data)
                    mask[np.triu_indices_from(mask)] = True
                    create_heatmap(csd_data, ch_names=csd.ch_names, mask=mask,
                                   title='Cross Spectral Density',
                                   title_item=f'{fmin}-{fmax}', unit='Hz')
                elif len(freq_band) == 1:
                    freq_band = freq_band[0]
                    csd_data = np.abs(csd.mean(freq_band, freq_band).get_data())
                    mask = np.zeros_like(csd_data)
                    mask[np.triu_indices_from(mask)] = True
                    create_heatmap(csd_data, ch_names=csd.ch_names, mask=mask,
                                   title='Cross Spectral Density',
                                   title_item=str(freq_band), unit='Hz')
            else:
                if len(freq_band) == 1:
                    freq_band = freq_band[0]
                    csd_data = np.abs(csd.get_data(frequency=freq_band))
                    mask = np.zeros_like(csd_data)
                    mask[np.triu_indices_from(mask)] = True
                    create_heatmap(csd_data, ch_names=csd.ch_names, mask=mask,
                                   title='Cross Spectral Density',
                                   title_item=str(freq_band), unit='Hz')
                else:
                    fmin = freq_band[0]
                    fmax = freq_band[1]
                    if fmin > fmax:
                        return
                    fmin_index = np.argwhere(self.freqs >= fmin)[0]
                    fmax_index = np.argwhere(self.freqs <= fmax)[-1]
                    freqs = self.freqs[fmin_index[0]: fmax_index[0]+1]
                    csd_data = []
                    for freq in freqs:
                        csd_data.append(csd.get_data(frequency=freq))
                    csd_data = np.abs(np.asarray(csd_data))
                    freqs = [str(round(freq, 3)) for freq in freqs]
                    mask = np.zeros_like(csd_data[0])
                    mask[np.triu_indices_from(mask)] = True
                    create_heatmap(csd_data, ch_names=csd.ch_names, mask=mask,
                                   title='Cross Spectral Density',
                                   title_item=freqs, unit='Hz')