from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("spike_time",
                 sources=["_spike_time.pyx", "spike_time.c"],
                 include_dirs=[numpy.get_include()])],
)