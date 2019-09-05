import tensorflow as tf
print(tf.sysconfig.get_lib(), tf.sysconfig.get_include(), tf.sysconfig.CXX11_ABI_FLAG, sep=";", end="")