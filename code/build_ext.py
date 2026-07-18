from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
exts=[Extension(n,[n+".pyx"],extra_compile_args=["-ffp-contract=off","-O3","-fno-fast-math"])
      for n in ("fast_interval_cy","g_aa_cy","g_da_cy","tm2_cy")]
setup(ext_modules=cythonize(exts,language_level=3,quiet=True),script_args=["build_ext","--inplace"])
