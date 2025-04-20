import gc
from tensorflow.keras import backend as K
import tensorflow as tf
import ast

def clear_tf_memory():
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

def parse_param_value(val, k, param_space):
    if isinstance(param_space[k][0], list):
        return tuple(ast.literal_eval(str(val)))
    elif isinstance(param_space[k][0], float):
        return float(val)
    elif isinstance(param_space[k][0], int):
        return int(val)
    else:
        return val
