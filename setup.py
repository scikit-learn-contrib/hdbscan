import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


class CustomBuildExtCommand(build_ext):
    def run(self):
        self.include_dirs.append(numpy.get_include())
        super().run()


extensions = [
    Extension("hdbscan._hdbscan_tree", ["hdbscan/_hdbscan_tree.pyx"]),
    Extension("hdbscan._hdbscan_linkage", ["hdbscan/_hdbscan_linkage.pyx"]),
    Extension("hdbscan._hdbscan_boruvka", ["hdbscan/_hdbscan_boruvka.pyx"]),
    Extension("hdbscan._hdbscan_reachability", ["hdbscan/_hdbscan_reachability.pyx"]),
    Extension("hdbscan._prediction_utils", ["hdbscan/_prediction_utils.pyx"]),
    Extension("hdbscan.dist_metrics", ["hdbscan/dist_metrics.pyx"]),
]

setup(
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": CustomBuildExtCommand},
)