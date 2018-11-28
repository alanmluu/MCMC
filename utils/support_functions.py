import tensorflow as tf


def gaussian_energy(x, mu, i_sigma):
    return tf.diag_part(0.5 * tf.matmul(tf.matmul(x - mu, i_sigma),
                                        tf.transpose(x - mu)))
