# -*- coding: UTF-8 -*-
"""
@Project ：EpiLocker 
@File    ：_locate_ieeg.py
@Author  ：Barry
@Date    ：2022/1/20 0:32 
"""
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal

from utils.decorator import safe_event
import mne.gui._ieeg_locate_gui as mne_ieeg


class SEEGLocator(mne_ieeg.IntracranialElectrodeLocator):
    CLOSE_SIGNAL = pyqtSignal(dict)

    def __init__(self, info, trans, aligned_ct, subject=None,
                 subjects_dir=None, groups=None, verbose=None):
        super(SEEGLocator, self).__init__(info, trans, aligned_ct, subject,
                                          subjects_dir, groups, verbose='error')

    @safe_event
    def closeEvent(self, event) -> None:
        self._renderer.plotter.close()
        self.CLOSE_SIGNAL.emit(self._chs)
        self.close()


def locate_ieeg(info, trans, aligned_ct, subject=None, subjects_dir=None,
                groups=None):
    """Locate intracranial electrode contacts.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    info
    aligned_ct : path-like | nibabel.spatialimages.SpatialImage
        The CT image that has been aligned to the Freesurfer T1. Path-like
        inputs and nibabel image objects are supported.
    %(subject)s
    %(subjects_dir)s
    groups : dict | None
        A dictionary with channels as keys and their group index as values.
        If None, the groups will be inferred by the channel names. Channel
        names must have a format like ``LAMY 7`` where a string prefix
        like ``LAMY`` precedes a numeric index like ``7``. If the channels
        are formatted improperly, group plotting will work incorrectly.
        Group assignments can be adjusted in the GUI.
    %(verbose)s

    Returns
    -------
    gui : instance of IntracranialElectrodeLocator
        The graphical user interface (GUI) window.
    """
    gui = SEEGLocator(info, trans, aligned_ct, subject=subject,
                      subjects_dir=subjects_dir, groups=groups)
    gui.setFont(QFont('Ubuntu'))
    gui.show()
    return gui
