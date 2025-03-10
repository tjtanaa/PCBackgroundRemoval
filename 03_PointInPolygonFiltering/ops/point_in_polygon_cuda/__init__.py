from pathlib import Path

import tensorflow as tf

pip_module = tf.load_op_library(str(Path(__file__).parent / 'build' / 'point_in_polygon.so'))
def pip_wna_number(points: tf.Tensor, vertices: tf.Tensor) -> tf.Tensor:
    """
    inputs:
        - `float32` tensor, shape `(npoints, 3)`
        - `float32` tensor, shape `(nvertices, 2)`

    returns:
        - `int32` tensor, shape `(npoints,)`
    """
    return pip_module.pipnumber_single_polygon(points, vertices)
tf.no_gradient('PipnumberSinglePolygon')
