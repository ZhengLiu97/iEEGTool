# -*- coding: UTF-8 -*-
'''
@Project ：EpiLock 
@File    ：_config.py
@Author  ：Barry
@Date    ：2021/11/15 21:20 
'''

# 20 colors generated to be evenly spaced in a cube, worked better than
# matplotlib color cycle
_mne_unipue_color = [(0.1, 0.42, 0.43), (0.9, 0.34, 0.62), (0.47, 0.51, 0.3),
                  (0.47, 0.55, 0.99), (0.79, 0.68, 0.06), (0.34, 0.74, 0.05),
                  (0.58, 0.87, 0.13), (0.86, 0.98, 0.4), (0.92, 0.91, 0.66),
                  (0.77, 0.38, 0.34), (0.9, 0.37, 0.1), (0.2, 0.62, 0.9),
                  (0.22, 0.65, 0.64), (0.14, 0.94, 0.8), (0.34, 0.31, 0.68),
                  (0.59, 0.28, 0.74), (0.46, 0.19, 0.94), (0.37, 0.93, 0.7),
                  (0.56, 0.86, 0.55), (0.67, 0.69, 0.44)]

color = ("#FF0000", "#EB8E55", "#CD853F", "#1E90FF", "#228B22",
         "#FF4500", "#0000FF", "#00FFFF", "#8A2BE2", "#D2691E",
         "#00FF00", "#4B0082", "#FF8C00", "#00C78C", "#ED9121",
         "#40E0D0", "#FF00FF", "#FFA500", "#8B4513", "#DC143C")


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