# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：freesurfer.py
@Author  ：Barry
@Date    ：2022/3/8 23:44 
"""
import numpy as np
import os.path as op

def read_freesurfer_lut(fname=None):
    """Read a Freesurfer-formatted LUT.
    Parameters
    ----------
    fname : str | None
        The filename. Can be None to read the standard Freesurfer LUT.
    Returns
    -------
    atlas_ids : dict
        Mapping from label names to IDs.
    colors : dict
        Mapping from label names to colors.
    """
    lut = _get_lut(fname)
    names, ids = lut['name'], lut['id']
    colors = np.array([lut['R'], lut['G'], lut['B'], lut['A']], float).T
    atlas_ids = dict(zip(names, ids))
    colors = dict(zip(names, colors))
    return atlas_ids, colors


def _get_lut(fname=None):
    """Get a FreeSurfer LUT."""
    if fname is None:
        fname = op.join(op.dirname(__file__), 'data',
                        'FreeSurferColorLUT.txt')
    dtype = [('id', '<i8'), ('name', 'U'),
             ('R', '<i8'), ('G', '<i8'), ('B', '<i8'), ('A', '<i8')]
    lut = {d[0]: list() for d in dtype}
    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            line = line.split()
            if len(line) != len(dtype):
                raise RuntimeError(f'LUT is improperly formatted: {fname}')
            for d, part in zip(dtype, line):
                lut[d[0]].append(part)
    lut = {d[0]: np.array(lut[d[0]], dtype=d[1]) for d in dtype}
    assert len(lut['name']) > 0
    return lut