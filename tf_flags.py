import tensorflow as tf

print(" ".join(tf.sysconfig.get_link_flags()), tf.sysconfig.get_include(), tf.sysconfig.CXX11_ABI_FLAG, sep=";", end="")
