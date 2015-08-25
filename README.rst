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

------------------
How to use HDBSCAN
------------------

The hdbscan package inherits from sklearn classes, and thus drops in neatly
next to other sklearn clusterers with an indentical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe, or
sparse matrix) of shape `(num_samples x num_features)`; an array (or sparse matrix)
giving a distance matrix between samples.

.. code:: python

    import hdbscan
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(data)

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
