import warnings

try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.message)
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext

_hdbscan_tree = Extension('hdbscan/_hdbscan_tree',
                          sources=['hdbscan/_hdbscan_tree.pyx'])
_hdbscan_linkage = Extension('hdbscan/_hdbscan_linkage',
                             sources=['hdbscan/_hdbscan_linkage.pyx'])

configuration = {
    'name' : 'hdbscan',
    'packages' : ['hdbscan'],
    'install_requires' : ['cython >= 0.17'],
    'ext_modules' : [_hdbscan_tree, _hdbscan_linkage],
    'cmdclass' : {'build_ext' : build_ext}
    }

if not HAVE_CYTHON:
    _hdbscan_tree.sources[0] = '_hdbscan_tree.c'
    _hdbscan_linkage.sources[0] = '_hdbscan_linkage.c'
    configuration.pop('install_requires')

setup(**configuration)
 
