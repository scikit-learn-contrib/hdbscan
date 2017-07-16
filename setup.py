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

import numpy

_hdbscan_tree = Extension('hdbscan._hdbscan_tree',
                          sources=['hdbscan/_hdbscan_tree.pyx'],
                          include_dirs=[numpy.get_include()])
_hdbscan_linkage = Extension('hdbscan._hdbscan_linkage',
                             sources=['hdbscan/_hdbscan_linkage.pyx'],
                             include_dirs=['hdbscan', numpy.get_include()])
_hdbscan_boruvka = Extension('hdbscan._hdbscan_boruvka',
                             sources=['hdbscan/_hdbscan_boruvka.pyx'],
                             include_dirs=['hdbscan', numpy.get_include()])
_hdbscan_reachability = Extension('hdbscan._hdbscan_reachability',
                                  sources=['hdbscan/_hdbscan_reachability.pyx'],
                                  include_dirs=[numpy.get_include()])
_prediction_utils = Extension('hdbscan._prediction_utils',
                              sources=['hdbscan/_prediction_utils.pyx'],
                              include_dirs=[numpy.get_include()])
dist_metrics = Extension('hdbscan.dist_metrics',
                         sources=['hdbscan/dist_metrics.pyx'],
                         include_dirs=[numpy.get_include()])

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'hdbscan',
    'version' : '0.8.11',
    'description' : 'Clustering based on density with variable density clusters',
    'long_description' : readme(),
    'classifiers' : [
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
    'keywords' : 'cluster clustering density hierarchical',
    'url' : 'http://github.com/scikit-learn-contrib/hdbscan',
    'maintainer' : 'Leland McInnes',
    'maintainer_email' : 'leland.mcinnes@gmail.com',
    'license' : 'BSD',
    'packages' : ['hdbscan'],
    'install_requires' : ['scikit-learn>=0.16',
                          'cython >= 0.17'],
    'ext_modules' : [_hdbscan_tree,
                     _hdbscan_linkage,
                     _hdbscan_boruvka,
                     _hdbscan_reachability,
                     _prediction_utils,
                     dist_metrics],
    'cmdclass' : {'build_ext' : build_ext},
    'test_suite' : 'nose.collector',
    'tests_require' : ['nose'],
    'data_files' : ('hdbscan/dist_metrics.pxd',)
    }

if not HAVE_CYTHON:
    _hdbscan_tree.sources[0] = 'hdbscan/_hdbscan_tree.c'
    _hdbscan_linkage.sources[0] = 'hdbscan/_hdbscan_linkage.c'
    _hdbscan_boruvka.sources[0] = 'hdbscan/_hdbscan_boruvka.c'
    _hdbscan_reachability.sources[0] = 'hdbscan/_hdbscan_reachability.c'
    _prediction_utils.sources[0] = 'hdbscan/_prediction_utils.c'
    dist_metrics.sources[0] = 'hdbscan/dist_metrics.c'
    configuration['install_requires'] = ['scikit-learn>=0.16']

setup(**configuration)
 
