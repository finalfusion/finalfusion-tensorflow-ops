import platform
import tensorflow as tf

l_flags = tf.sysconfig.get_link_flags()
if platform.system() == "Darwin":
    l_flags[1] = "-l"+l_flags[1].strip(".dylib").strip("-l:lib")
print(" ".join(l_flags), tf.sysconfig.get_include(), tf.sysconfig.CXX11_ABI_FLAG, sep=";", end="")
