# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：subject.py
@Author  ：Barry
@Date    ：2022/2/18 1:52 
"""
class Subject(object):

    def __init__(self, name=None):
        self._name = name
        self._t1 = None
        self._ct = None
        self._align_ct = None
        self._ieeg = None
        self._electrodes = None

    def set_name(self, name):
        self._name = name

    def set_t1(self, t1):
        self._t1 = t1

    def set_ct(self, ct):
        self._ct = ct

    def set_align_ct(self, _align_ct):
        self._align_ct = _align_ct

    def set_ieeg(self, ieeg):
        self._ieeg = ieeg

    def set_electrodes(self, electrodes):
        self._electrodes = electrodes

    def get_name(self):
        return self._name

    def get_t1(self):
        return self._t1

    def get_ct(self):
        return self._ct

    def get_align_ct(self):
        return self._align_ct

    def get_ieeg(self):
        return self._ieeg

    def get_electrodes(self):
        return self._electrodes

    def remove_name(self):
        self._name = None

    def remove_t1(self):
        self._t1 = None

    def remove_ct(self):
        self._ct = None

    def remove_align_ct(self):
        self._align_ct = None

    def remove_ieeg(self):
        self._ieeg = None

    def remove_electrodes(self):
        self._electrodes = None


a = Subject()