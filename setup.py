import warnings

try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    HAVE_CYTHON = False


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


_hdbscan_tree = Extension('hdbscan._hdbscan_tree',
                          sources=['hdbscan/_hdbscan_tree.pyx'])
_hdbscan_linkage = Extension('hdbscan._hdbscan_linkage',
                             sources=['hdbscan/_hdbscan_linkage.pyx'])
_hdbscan_boruvka = Extension('hdbscan._hdbscan_boruvka',
                             sources=['hdbscan/_hdbscan_boruvka.pyx'])
_hdbscan_reachability = Extension('hdbscan._hdbscan_reachability',
                                  sources=['hdbscan/_hdbscan_reachability.pyx'])
_prediction_utils = Extension('hdbscan._prediction_utils',
                              sources=['hdbscan/_prediction_utils.pyx'])
dist_metrics = Extension('hdbscan.dist_metrics',
                         sources=['hdbscan/dist_metrics.pyx'])


def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()


configuration = {
    'name': 'hdbscan',
    'version': '0.8.13',
    'description': 'Clustering based on density with variable density clusters',
    'long_description': readme(),
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    'keywords': 'cluster clustering density hierarchical',
    'url': 'http://github.com/scikit-learn-contrib/hdbscan',
    'maintainer': 'Leland McInnes',
    'maintainer_email': 'leland.mcinnes@gmail.com',
    'license': 'BSD',
    'packages': ['hdbscan', 'hdbscan.tests'],
    'install_requires': ['numpy',
                         'scikit-learn>=0.16',
                         'cython >= 0.17'],
    'ext_modules': [_hdbscan_tree,
                    _hdbscan_linkage,
                    _hdbscan_boruvka,
                    _hdbscan_reachability,
                    _prediction_utils,
                    dist_metrics],
    'cmdclass': {'build_ext': CustomBuildExtCommand},
    'test_suite': 'nose.collector',
    'tests_require': ['nose'],
    'data_files': ('hdbscan/dist_metrics.pxd',)
}

if not HAVE_CYTHON:
    _hdbscan_tree.sources[0] = 'hdbscan/_hdbscan_tree.c'
    _hdbscan_linkage.sources[0] = 'hdbscan/_hdbscan_linkage.c'
    _hdbscan_boruvka.sources[0] = 'hdbscan/_hdbscan_boruvka.c'
    _hdbscan_reachability.sources[0] = 'hdbscan/_hdbscan_reachability.c'
    _prediction_utils.sources[0] = 'hdbscan/_prediction_utils.c'
    dist_metrics.sources[0] = 'hdbscan/dist_metrics.c'
    configuration['install_requires'] = ['numpy', 'scikit-learn>=0.16']

setup(**configuration)
