import os
import platform
import subprocess
import sys

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

if platform.system() == "Darwin":
    LIB_SUFFIX = ".dylib"
else:
    LIB_SUFFIX = ".so"


# this method is inspired by horovod's setup.py which can be found at:
# https://github.com/horovod/horovod/blob/master/setup.py
def get_compiler_version():
    if sys.platform.startswith('linux') and not os.getenv('CC') and not os.getenv('CXX'):
        import tensorflow as tf
        if hasattr(tf, 'version'):
            # Since TensorFlow 1.13.0
            tf_compiler_version = LooseVersion(tf.version.COMPILER_VERSION)
        else:
            tf_compiler_version = LooseVersion(tf.COMPILER_VERSION)

        if tf_compiler_version.version[0] == 4:
            # https://github.com/tensorflow/tensorflow/issues/27067
            maximum_compiler_version = LooseVersion('5')
        else:
            maximum_compiler_version = LooseVersion('999')

        if len(tf_compiler_version.version) >= 3:
            tf_compiler_version = LooseVersion(
                str(tf_compiler_version.version[0]) + "." + str(tf_compiler_version.version[1]))
        compiler_version = LooseVersion('0')
        cur_gcc_path = cur_gxx_path = None
        # iterate over binaries on PATH
        for directory in os.getenv('PATH', '').split(':'):
            if not os.path.exists(directory):
                continue
            for bin in os.listdir(directory):
                if bin.startswith("g++"):
                    gxx_path = os.path.join(directory, bin)
                    gxx_cmd = "{0} -dumpversion".format(gxx_path)
                    gxx_ver = LooseVersion(subprocess.check_output(gxx_cmd, shell=True,
                                                                   universal_newlines=True).strip())
                    # check if gxx version is compatible
                    if maximum_compiler_version > gxx_ver >= tf_compiler_version:
                        gcc_path = os.path.join(directory, "gcc" + bin[3:])
                        # verify gcc with same version to g++ exists
                        if os.path.exists(gcc_path):
                            gcc_ver_cmd = "{0} -dumpversion".format(gcc_path)
                            gcc_ver = LooseVersion(
                                subprocess.check_output(gcc_ver_cmd, shell=True, universal_newlines=True).strip())
                            # verify that gcc and g++ match and version is higher than currently highest
                            if gcc_ver == gxx_ver > compiler_version:
                                cur_gcc_path, cur_gxx_path = gcc_path, gxx_path
        if cur_gxx_path is None:
            raise RuntimeError(
                "Can't find suitable g++ compiler, max version: {0}, min version {1}".format(maximum_compiler_version,
                                                                                             tf_compiler_version))
        return cur_gcc_path, cur_gxx_path
    else:
        return None


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    user_options = [('test', None, 'Test after building the extension.'), ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.test = False

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        self.build_extensions()

    def build_extensions(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            cfg = 'Debug' if self.debug else 'Release'

            cmake_args = [
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), self.build_temp),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
            ]

            compilers = get_compiler_version()
            if compilers:
                os.environ['CC'], os.environ['CXX'] = compilers
                print("Compiling with: {0}".format(os.environ['CXX']))
            # ensure CMake calls same Python exectuable
            os.environ['PYTHON_EXECUTABLE'] = sys.executable
            self.mkpath(self.build_temp)

            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.', '--config', cfg],
                                  cwd=self.build_temp)

            op_dir = os.path.join(os.path.abspath(self.build_lib), ext.name, "ops")
            self.mkpath(op_dir)
            self.copy_file(os.path.join(self.build_temp, "finalfusion-tf", "libfinalfusion_tf" + LIB_SUFFIX), op_dir)
            if self.test:
                subprocess.check_call(['ctest', '-V'], cwd=self.build_temp)


setup(
    author='The finalfusion authors',
    author_email='sebastian.puetz@uni-tuebingen.de',
    cmdclass=dict(build_ext=cmake_build_ext),
    description='Tensorflow Ops for Finalfusion embeddings.',
    ext_modules=[CMakeExtension(name='finalfusion_tensorflow')],
    license='BlueOak-1.0.0',
    name="finalfusion_tensorflow",
    packages=find_packages(where="./python"),
    package_dir={
        '': 'python',
    },
    url='https://github.com/finalfusion/finalfusion-tensorflow',
    version='0.1.0',
)
