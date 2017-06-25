.. image:: https://img.shields.io/pypi/v/hdbscan.svg
    :target: https://pypi.python.org/pypi/hdbscan/
    :alt: PyPI Version
.. image:: https://anaconda.org/conda-forge/hdbscan/badges/version.svg
    :target: https://anaconda.org/conda-forge/hdbscan
    :alt: Conda-forge Version
.. image:: https://anaconda.org/conda-forge/hdbscan/badges/downloads.svg
    :target: https://anaconda.org/conda-forge/hdbscan
    :alt: Conda-forge downloads
.. image:: https://img.shields.io/pypi/l/hdbscan.svg
    :target: https://github.com/scikit-learn-contrib/hdbscan/blob/master/LICENSE
    :alt: License
.. image:: https://travis-ci.org/scikit-learn-contrib/hdbscan.svg
    :target: https://travis-ci.org/scikit-learn-contrib/hdbscan
    :alt: Travis Build Status
.. image:: https://coveralls.io/repos/github/scikit-learn-contrib/hdbscan/badge.svg?branch=master
    :target: https://coveralls.io/github/scikit-learn-contrib/hdbscan?branch=master
    :alt: Test Coverage
.. image:: https://readthedocs.org/projects/hdbscan/badge/?version=latest
    :target: https://hdbscan.readthedocs.org
    :alt: Docs
.. image:: http://joss.theoj.org/papers/10.21105/joss.00205/status.svg
    :target: http://joss.theoj.org/papers/10.21105/joss.00205
    :alt: JOSS article


=======
HDBSCAN
=======

HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
with Noise. Performs DBSCAN over varying epsilon values and integrates 
the result to find a clustering that gives the best stability over epsilon.
This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
and be more robust to parameter selection.

In practice this means that HDBSCAN returns a good clustering straight
away with little or no parameter tuning -- and the primary parameter,
minimum cluster size, is intuitive and easy to select.

HDBSCAN is ideal for exploratory data analysis; it's a fast and robust
algorithm that you can trust to return meaningful clusters (if there
are any).

Based on the paper:
    R. Campello, D. Moulavi, and J. Sander, *Density-Based Clustering Based on
    Hierarchical Density Estimates*
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    
Documentation, including tutorials, are available on ReadTheDocs at http://hdbscan.readthedocs.io/en/latest/ .  
    
Notebooks `comparing HDBSCAN to other clustering algorithms <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb>`_, explaining `how HDBSCAN works <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ and `comparing performance with other python clustering implementations <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.7.ipynb>`_ are available.

------------------
How to use HDBSCAN
------------------

The hdbscan package inherits from sklearn classes, and thus drops in neatly
next to other sklearn clusterers with an identical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe, or
sparse matrix) of shape ``(num_samples x num_features)``; an array (or sparse matrix)
giving a distance matrix between samples.

.. code:: python

    import hdbscan
    from sklearn.datasets import make_blobs
    
    data, _ = make_blobs(1000)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(data)

-----------
Performance
-----------

Significant effort has been put into making the hdbscan implementation as fast as 
possible. It is `orders of magnitude faster than the reference implementation <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Python%20vs%20Java.ipynb>`_ in Java,
and is currently faster than highly optimized single linkage implementations in C and C++.
`version 0.7 performance can be seen in this notebook <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.7.ipynb>`_ .
In particular `performance on low dimensional data is better than sklearn's DBSCAN <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations%202D%20v0.7.ipynb>`_ ,
and via support for caching with joblib, re-clustering with different parameters
can be almost free.

------------------------
Additional functionality
------------------------

The hdbscan package comes equipped with visualization tools to help you
understand your clustering results. After fitting data the clusterer
object has attributes for:

* The condensed cluster hierarchy
* The robust single linkage cluster hierarchy
* The reachability distance minimal spanning tree

All of which come equipped with methods for plotting and converting
to Pandas or NetworkX for further analysis. See the notebook on
`how HDBSCAN works <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ for examples and further details.

The clusterer objects also have an attribute providing cluster membership
strengths, resulting in optional soft clustering (and no further compute 
expense). Finally each cluster also receives a persistence score giving
the stability of the cluster over the range of distance scales present
in the data. This provides a measure of the relative strength of clusters.

-----------------
Outlier Detection
-----------------

The HDBSCAN clusterer objects also support the GLOSH outlier detection algorithm. 
After fitting the clusterer to data the outlier scores can be accessed via the
``outlier_scores_`` attribute. The result is a vector of score values, one for
each data point that was fit. Higher scores represent more outlier like objects.
Selecting outliers via upper quantiles is often a good approach.

Based on the paper:
    R.J.G.B. Campello, D. Moulavi, A. Zimek and J. Sander 
    *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection*, 
    ACM Trans. on Knowledge Discovery from Data, Vol 10, 1 (July 2015), 1-51.

---------------------
Robust single linkage
---------------------

The hdbscan package also provides support for the *robust single linkage*
clustering algorithm of Chaudhuri and Dasgupta. As with the HDBSCAN 
implementation this is a high performance version of the algorithm 
outperforming scipy's standard single linkage implementation. The
robust single linkage hierarchy is available as an attribute of
the robust single linkage clusterer, again with the ability to plot
or export the hierarchy, and to extract flat clusterings at a given
cut level and gamma value.

Example usage:

.. code:: python

    import hdbscan
    from sklearn.datasets import make_blobs
    
    data = make_blobs(1000)
    
    clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
    cluster_labels = clusterer.fit_predict(data)
    hierarchy = clusterer.cluster_hierarchy_
    alt_labels = hierarchy.get_clusters(0.100, 5)
    hierarchy.plot()


Based on the paper:
    K. Chaudhuri and S. Dasgupta.
    *"Rates of convergence for the cluster tree."*
    In Advances in Neural Information Processing Systems, 2010.

----------
Installing
----------

Easiest install, if you have Anaconda (thanks to conda-forge which is awesome!):

.. code:: bash

    conda install -c conda-forge hdbscan

PyPI install, presuming you have sklearn and all its requirements (numpy and scipy) installed:

.. code:: bash

    pip install hdbscan

If pip is having difficulties pulling the dependencies then we'd suggest installing
the dependencies manually using anaconda followed by pulling hdbscan from pip:

.. code:: bash

    conda install cython
    conda install numpy scipy
    conda install scikit-learn
    pip install hdbscan

For a manual install get this package:

.. code:: bash

    wget https://github.com/scikit-learn-contrib/hdbscan/archive/master.zip
    unzip master.zip
    rm master.zip
    cd hdbscan-master

Install the requirements

.. code:: bash

    sudo pip install -r requirements.txt
    
or

.. code:: bash

    conda install scikit-learn cython 

Install the package

.. code:: bash

    python setup.py install
    
--------------
Python Version
--------------

The hdbscan library supports both Python 2 and Python 3. However we recommend Python 3 as the better option if it is available to you.
    
----------------
Help and Support
----------------

For simple issues you can consult the `FAQ <https://hdbscan.readthedocs.io/en/latest/faq.html>`_ in the documentation.
If your issue is not suitably resolved there, please check the `issues <https://github.com/scikit-learn-contrib/hdbscan/issues>`_ on github. Finally, if no solution is available there feel free to `open an issue <https://github.com/scikit-learn-contrib/hdbscan/issues/new>`_ ; the authors will attempt to respond in a reasonably timely fashion.

------------
Contributing
------------

We welcome contributions in any form! Assistance with documentation, particularly expanding tutorials,
is always welcome. To contribute please `fork the project <https://github.com/scikit-learn-contrib/hdbscan/issues#fork-destination-box>`_ make your changes and submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.

------
Citing
------

If you have used this codebase in a scientific publication and wish to cite it, please use the `Journal of Open Source Software article <http://joss.theoj.org/papers/10.21105/joss.00205>`_.

    L. McInnes, J. Healy, S. Astels, *hdbscan: Hierarchical density based clustering*
    In: Journal of Open Source Software, The Open Journal, volume 2, number 11.
    2017

---------
Licensing
---------

The hdbscan package is 3-clause BSD licensed. Enjoy.
