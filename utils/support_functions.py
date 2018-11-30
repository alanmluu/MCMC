import tensorflow as tf
import matplotlib.pyplot as plt

COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

def gaussian_energy(x, mu, i_sigma):
    return tf.diag_part(0.5 * tf.matmul(tf.matmul(x - mu, i_sigma),
                                        tf.transpose(x - mu)))
