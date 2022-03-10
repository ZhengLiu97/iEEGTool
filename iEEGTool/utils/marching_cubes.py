# -*- coding: UTF-8 -*-
"""
@Project ：EpiLock 
@File    ：_mne_marching_cubes.py
@Author  ：Barry
@Date    ：2021/11/16 22:24 
"""
import numpy as np
from vtk import (vtkImageData, vtkThreshold,
                 vtkWindowedSincPolyDataFilter, vtkDiscreteFlyingEdges3D,
                 vtkGeometryFilter, vtkDataSetAttributes, VTK_DOUBLE)


def marching_cubes(image, level, smooth=0):
    """
    Compute marching cubes on a 3D image.
    :param image: volumn
    :param level: the region id
    :param smooth: smooth value
    :return:
    """
    from vtk.util import numpy_support
    smooth = float(smooth)
    if not 0 <= smooth < 1:
        raise ValueError('smooth must be between 0 (inclusive) and 1 '
                         f'(exclusive), got {smooth}')

    if image.ndim != 3:
        raise ValueError(f'3D data must be supplied, got {image.shape}')
    # force double as passing integer types directly can be problematic!
    image_shape = image.shape
    data_vtk = numpy_support.numpy_to_vtk(
        image.ravel(), deep=True, array_type=VTK_DOUBLE)
    del image
    level = np.array(level)
    if level.ndim != 1 or level.size == 0 or level.dtype.kind not in 'ui':
        raise TypeError(
            'level must be non-empty numeric or 1D array-like of int, '
            f'got {level.ndim}D array-like of {level.dtype} with '
            f'{level.size} elements')
    mc = vtkDiscreteFlyingEdges3D()
    # create image
    imdata = vtkImageData()
    imdata.SetDimensions(image_shape)
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(data_vtk)

    # compute marching cubes
    mc.SetNumberOfContours(len(level))
    for li, lev in enumerate(level):
        mc.SetValue(li, lev)
    mc.SetInputData(imdata)
    sel_input = mc
    if smooth:
        smoother = vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(100)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(120.0)
        smoother.SetPassBand(1 - smooth)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOff()
        # smoother.NormalizeCoordinatesOn()  # 二者结果没差别
        sel_input = smoother
    sel_input.Update()

    # get vertices and triangles
    selector = vtkThreshold()
    selector.SetInputConnection(sel_input.GetOutputPort())
    dsa = vtkDataSetAttributes()
    selector.SetInputArrayToProcess(
        0, 0, 0, imdata.FIELD_ASSOCIATION_POINTS, dsa.SCALARS)
    geometry = vtkGeometryFilter()
    geometry.SetInputConnection(selector.GetOutputPort())
    out = list()
    index = list()
    for val in level:
        selector.ThresholdBetween(val, val)
        geometry.Update()
        polydata = geometry.GetOutput()
        # rr = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        try:
            rr = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
            tris = numpy_support.vtk_to_numpy(
                polydata.GetPolys().GetConnectivityArray()).reshape(-1, 3)
            rr = np.ascontiguousarray(rr[:, ::-1])
            tris = np.ascontiguousarray(tris[:, ::-1])
            out.append((rr, tris))
        except:
            index.append(val)

    return out, index
