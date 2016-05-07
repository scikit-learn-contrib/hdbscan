.. image:: https://img.shields.io/pypi/v/hdbscan.svg
    :target: https://pypi.python.org/pypi/hdbscan/
    :alt: Version
.. image:: https://img.shields.io/pypi/l/hdbscan.svg
    :target: https://github.com/nicodv/hdbscan/blob/master/LICENSE
    :alt: License

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
    
Notebooks `comparing HDBSCAN to other clustering algorithms <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb>`_, explaining `how HDBSCAN works <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ and `comparing performance with other python clustering implementations <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.7.ipynb>`_ are available.

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
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(data)

-----------
Performance
-----------

Significant effort has been put into making the hdbscan implementation as fast as 
possible. It is `orders of magnitude faster than the reference implementation <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Python%20vs%20Java.ipynb>`_ in Java,
and is currently faster than highly optimized single linkage implementations in C and C++.
`version 0.7 performance can be seen in this notebook <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.7.ipynb>`_ .
In particular `performance on low dimensional data is better than sklearn's DBSCAN <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations%202D%20v0.7.ipynb>`_ ,
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
`how HDBSCAN works <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ for examples and further details.

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

Fast install, presuming you have sklearn and all its requirements installed:

.. code:: bash

    pip install hdbscan

If pip is having difficulties pulling the dependencies then we'd suggest installing
the dependencies manually using anaconda followed by pulling hdbscan from pip:

.. code:: bash

    conda install cython
    conda install scikit-learn
    pip install hdbscan

For a manual install get this package:

.. code:: bash

    wget https://github.com/lmcinnes/hdbscan/archive/master.zip
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

Coming soon: installing via conda.

---------
Licensing
---------

The hdbscan package is 3-clause BSD licensed. Enjoy.
