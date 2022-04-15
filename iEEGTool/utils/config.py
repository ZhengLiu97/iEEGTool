# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool
@File    ：_config.py
@Author  ：Barry
@Date    ：2021/11/15 21:20
"""

color = ("#FF0000", "#EB8E55", "#CD853F", "#1E90FF", "#228B22",
         "#FF4500", "#0000FF", "#00FFFF", "#8A2BE2", "#D2691E",
         "#00FF00", "#4B0082", "#FF8C00", "#00C78C", "#ED9121",
         "#40E0D0", "#FF00FF", "#FFA500", "#8B4513", "#DC143C")

brain_kwargs = dict(
    color=(0.9, 0.9, 0.9), style='surface',
    opacity=0.1, lighting=True, ambient=0.4225,
    specular=0.3, specular_power=20, diffuse=0.5,
    line_width=10., smooth_shading=True, reset_camera=True
)

contact_kwargs = dict(
    style='surface', lighting=True, ambient=0.4225,
    specular=0.3, specular_power=20, diffuse=0.5,
    smooth_shading=True, line_width=1., reset_camera=False
)

text_kwargs = dict(
    font_size=12, text_color='k', show_points=False,
    shape_opacity=0., tolerance=0.1, reset_camera=False
)

roi_kwargs = dict(
    style='surface', opacity=0.6, render_points_as_spheres=True,
    lighting=True, smooth_shading=True, line_width=1.,
    reset_camera=False
)

view_dict = {
    'front': [(0, 1, 0), (0, 0, 1)], 'back': [(0, -1, 0), (0, 0, 1)],
    'left': [(-1, 0, 0), (0, 0, 1)], 'right': [(1, 0, 0), (0, 0, 1)],
    'top': [(0, 0, 1), (0, 1, 0)], 'bottom': [(0, 0, -1), (0, 1, 0)],
        }

def _to_rgb(*args, name='color', alpha=False):
    from matplotlib.colors import colorConverter
    func = colorConverter.to_rgba if alpha else colorConverter.to_rgb
    try:
        return func(*args)
    except ValueError:
        args = args[0] if len(args) == 1 else args
        raise ValueError(
            f'Invalid RGB{"A" if alpha else ""} argument(s) for {name}: '
            f'{repr(args)}') from None

def rgb2hex(colortuple):
    return '#' + ''.join(f'{i:02X}' for i in colortuple)