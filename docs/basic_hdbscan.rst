
Basic Usage of HDBSCAN\* for Clustering
=======================================

We have some data, and we want to cluster it. How exactly do we do that,
and what do the results look like? If you are very familiar with sklearn
and its API, particularly for clustering, then you can probably skip
this tutorial -- ``hdbscan`` implements exactly this API, so you can use
it just as you would any other sklearn clustering algorithm. If, on the
other hand, you aren't that familiar with sklearn, fear not, and read
on. Let's start with the simplest case first -- we have data in a nice
tidy dataframe format.

The Simple Case
---------------

Let's generate some data with, say 2000 samples, and 10 features. We can
put it in a dataframe for a nice clean table view of it.

.. code:: python

    from sklearn.datasets import make_blobs
    import pandas as pd
    
.. code:: python

    blobs, labels = make_blobs(n_samples=2000, n_features=10)

.. code:: python

    pd.DataFrame(blobs).head()


.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-3.370804</td>
          <td>8.487688</td>
          <td>4.631243</td>
          <td>-10.181475</td>
          <td>9.146487</td>
          <td>-8.070935</td>
          <td>-1.612017</td>
          <td>-2.418106</td>
          <td>-8.975390</td>
          <td>-1.769952</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-4.092931</td>
          <td>8.409841</td>
          <td>3.362516</td>
          <td>-9.748945</td>
          <td>9.556615</td>
          <td>-9.240307</td>
          <td>-2.038291</td>
          <td>-3.129068</td>
          <td>-7.109673</td>
          <td>-0.993827</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-4.604753</td>
          <td>9.616391</td>
          <td>4.631508</td>
          <td>-11.166361</td>
          <td>10.888212</td>
          <td>-8.427564</td>
          <td>-3.929517</td>
          <td>-4.563951</td>
          <td>-8.886373</td>
          <td>-1.995063</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-6.889866</td>
          <td>-7.801482</td>
          <td>-6.974958</td>
          <td>-8.570025</td>
          <td>5.438101</td>
          <td>-5.097457</td>
          <td>-4.941206</td>
          <td>-5.926394</td>
          <td>-10.145152</td>
          <td>0.219269</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.339728</td>
          <td>2.791309</td>
          <td>0.611464</td>
          <td>-2.929875</td>
          <td>-7.694973</td>
          <td>7.776050</td>
          <td>-1.218101</td>
          <td>0.408141</td>
          <td>-4.563975</td>
          <td>-1.309128</td>
        </tr>
      </tbody>
    </table>
    </div>



So now we need to import the hdbscan library.

.. code:: python

    import hdbscan

Now, to cluster we need to generate a clustering object.

.. code:: python

    clusterer = hdbscan.HDBSCAN()

We can then use this clustering object and fit it to the data we have.
This will return the clusterer object back to you -- just in case you
want do some method chaining.

.. code:: python

    clusterer.fit(blobs)


.. parsed-literal::

    HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
        metric='euclidean', min_cluster_size=5, min_samples=None, p=None)



At this point we are actually done! We've done the clustering! But where
are the results? How do I get the clusters? The clusterer object knows,
and stores the result in an attribute ``labels_``.

.. code:: python

    clusterer.labels_


.. parsed-literal::

    array([2, 2, 2, ..., 2, 2, 0])



So it is an array of integers. What are we to make of that? It is an
array with an integer for each data sample. Samples that are in the same
cluster get assigned the same number. The cluster labels start at 0 and count
up. We can thus determine the number of clusters found by finding the largest
cluster label.

.. code:: python

    clusterer.labels_.max()


.. parsed-literal::

    2

So we have a total of three clusters, with labels 0, 1, and 2.
Importantly HDBSCAN is noise aware -- it has a notion of data samples
that are not assigned to any cluster. This is handled by assigning these
samples the label -1. But wait, there's more. The ``hdbscan`` library
implements soft clustering, where each data point is assigned a cluster
membership score ranging from 0.0 to 1.0. A score of 0.0 represents a
sample that is not in the cluster at all (all noise points will get this
score) while a score of 1.0 represents a sample that is at the heart of
the cluster (note that this is not the spatial centroid notion of core).
You can access these scores via the ``probabilities_`` attribute.

.. code:: python

    clusterer.probabilities_


.. parsed-literal::

    array([ 0.83890858,  1.        ,  0.72629904, ...,  0.79456452,
            0.65311137,  0.76382928])



What about different metrics?
-----------------------------

That is all well and good, but even data that is embedded in a vector
space may not want to consider distances between data points to be pure
Euclidean distance. What can we do in that case? We are still in good
shape, since ``hdbscan`` supports a wide variety of metrics, which you
can set when creating the clusterer object. For example we can do the
following:

.. code:: python

    clusterer = hdbscan.HDBSCAN(metric='manhattan')
    clusterer.fit(blobs)
    clusterer.labels_




.. parsed-literal::

    array([1, 1, 1, ..., 1, 1, 0])



What metrics are supported? Because we simply steal metric computations
from sklearn we get a large number of metrics readily available.

.. code:: python

    hdbscan.dist_metrics.METRIC_MAPPING




.. parsed-literal::

    {'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
     'canberra': hdbscan.dist_metrics.CanberraDistance,
     'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
     'cityblock': hdbscan.dist_metrics.ManhattanDistance,
     'dice': hdbscan.dist_metrics.DiceDistance,
     'euclidean': hdbscan.dist_metrics.EuclideanDistance,
     'hamming': hdbscan.dist_metrics.HammingDistance,
     'haversine': hdbscan.dist_metrics.HaversineDistance,
     'infinity': hdbscan.dist_metrics.ChebyshevDistance,
     'jaccard': hdbscan.dist_metrics.JaccardDistance,
     'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
     'l1': hdbscan.dist_metrics.ManhattanDistance,
     'l2': hdbscan.dist_metrics.EuclideanDistance,
     'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
     'manhattan': hdbscan.dist_metrics.ManhattanDistance,
     'matching': hdbscan.dist_metrics.MatchingDistance,
     'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
     'p': hdbscan.dist_metrics.MinkowskiDistance,
     'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
     'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
     'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
     'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
     'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
     'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
     'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}



Distance matrices
-----------------

What if you don't have a nice set of points in a vector space, but only
have a pairwise distance matrix providing the distance between each pair
of points? This is a common situation. Perhaps you have a complex custom
distance measure; perhaps you have strings and are using Levenstein
distance, etc. Again, this is all fine as ``hdbscan`` supports a special
metric called ``precomputed``. If you create the clusterer with the
metric set to ``precomputed`` then the clusterer will assume that,
rather than being handed a vector of points in a vector space, it is
recieving an all pairs distance matrix. Missing distances can be
indicated by ``numpy.inf``, which leads HDBSCAN to ignore these pairwise
relationships as long as there exists a path between two points that
contains defined distances (i.e. if there are too many distances
missing, the clustering is going to fail).

NOTE: The input vector _must_ contain numerical data. If you have a 
distance matrix for non-numerical vectors, you will need to map your
input vectors to numerical vectors. (e.g use map ['A', 'G', 'C', 'T']->
[ 1, 2, 3, 4] to replace input vector ['A', 'A', 'A', 'C', 'G'] with
[ 1, 1, 1, 3, 2])

.. code:: python

    from sklearn.metrics.pairwise import pairwise_distances

.. code:: python

    distance_matrix = pairwise_distances(blobs)
    clusterer = hdbscan.HDBSCAN(metric='precomputed')
    clusterer.fit(distance_matrix)
    clusterer.labels_




.. parsed-literal::

    array([1, 1, 1, ..., 1, 1, 2])



Note that this result only appears different due to a different
labelling order for the clusters.

