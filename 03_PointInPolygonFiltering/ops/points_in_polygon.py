import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

from .point_in_polygon_cuda import pip_wna_number

def pip_wna_mask(
    points: np.ndarray,
    polygon_vertices: np.ndarray,
    inside: bool = False,
) -> np.ndarray:
    points = tf.constant(points, dtype=tf.float32)
    polygon_vertices = tf.constant(polygon_vertices, dtype=tf.float32)

    wn = pip_wna_number(points, polygon_vertices).numpy()

    if inside:
        return ~(wn == 0)
    else:
        return wn == 0
