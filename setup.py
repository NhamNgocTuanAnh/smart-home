from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

ext = Extension(name="module", sources=["module.pyx"])
setup(ext_modules=cythonize(ext, language_level = "3"))



# setup(
#     ext_modules=[
#         Extension("module", ["module.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

# setup(
#     ext_modules=cythonize("module.pyx"),
#     include_dirs=[numpy.get_include()]
# )