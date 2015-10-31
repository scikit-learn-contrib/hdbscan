=======
HDBSCAN
=======

HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
with Noise. Performs DBSCAN over varying epsilon values and integrates 
the result to find a clustering that gives the best stability over epsilon.
This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
and be more robust to parameter selection.

Based on the paper:
    R. Campello, D. Moulavi, and J. Sander, *Density-Based Clustering Based on
    Hierarchical Density Estimates*
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    
Notebooks `comparing HDBSCAN to other clustering algorithms <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb>`_, explaining `how HDBSCAN works <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ and `comparing performance with other python clustering implementations <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations.ipynb>`_ are available.

------------------
How to use HDBSCAN
------------------

The hdbscan package inherits from sklearn classes, and thus drops in neatly
next to other sklearn clusterers with an identical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe, or
sparse matrix) of shape `(num_samples x num_features)`; an array (or sparse matrix)
giving a distance matrix between samples.

.. code:: python

    import hdbscan
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(data)

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
expense)

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

Based on the paper:
    K. Chaudhuri and S. Dasgupta.
    *"Rates of convergence for the cluster tree."*
    In Advances in Neural Information Processing Systems, 2010.

----------
Installing
----------

Fast install

.. code:: bash

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

Install the package

.. code:: bash

    python setup.py install

---------
Licensing
---------

The hdbscan package is BSD licensed. Enjoy.
