import tensorflow as tf
import numpy as np

def unif_initializer(min_val, max_val):
    def _initializer(shape, dtype=None, partition_info=None):
        out = tf.random_uniform(shape, minval = min_val, maxval = max_val)
        return out
    return _initializer

def collection_add(list):
    for x in list:
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)

def clip_grads(grads, clip_val=10):
    clipped_gradients = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in grads if grad is not None]
    return clipped_gradients

def standardize_image(image):
    img_mean = np.mean(image)
    img_std = np.std(image)
    adj_std = np.maximum(img_std, 1. / np.sqrt(image.size))
    return (image - img_mean) / adj_std

def concat(state, goal):
    return np.concatenate([np.expand_dims(state, axis=1), np.expand_dims(goal, axis=1)], axis=1)
