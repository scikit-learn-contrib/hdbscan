
Extracting DBSCAN* clustering from HDBSCAN*
===========================================

There are a number of reasons that one might prefer `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`__'s
clustering over that of HDBSCAN*.  The biggest difficulty many folks have with
DBSCAN is that the epsilon distance parameter can be hard to determine and often
requires a great deal of trial and error to tune.  If your data lived in a more
interpretable space and you had a good notion of distance in that space this problem
is certainly mitigated and a user might want to set a very specific epsilon distance
for their use case.  Another viable use case might be that a user is interested in a
constant density clustering.
HDBSCAN* does variable density clustering by default, looking for the clusters that persist
over a wide range of epsilon distance parameters to find a 'natural' clustering.  This might
not be the right result for your application.  A DBSCAN clustering at a particular
epsilon value might work better for your particular task.

HDBSCAN returns a very natural clustering of your data which is often very useful in exploring
a new data set.  That doesn't necessarily make it the right clustering algorithm or every
task.

HDBSCAN* can best be thought of as a DBSCAN* implementation which varies across
all epsilon values and extracts the clusters that persist over the widest range
of these parameter choices.  It is therefore able to ignore the parameter and
only needs the minimum cluster size as single input parameter.
The 'eom' (Excess of Mass) cluster selection method then returns clusters with the
best stability over epsilon.

There are a number of alternative ways of extracting a flat clustering from
the HDBSCAN* hierarchical tree.  If one is interested in finer resolution
clusters while still maintaining variable density one could set
``cluster_selection_method='leaf'`` to extract the leaves of the condensed
tree instead of the most persistent clusters.  For more details on these
cluster selection methods see :ref:`leaf_clustering_label`.

If one wasn't interested in the variable density clustering that is the hallmark of
HDBSCAN* it is relatively easy to extract any DBSCAN* clustering from a
single run of HDBSCAN*.  This has the advantage of allowing you to perform
a single computationally efficient HDBSCAN* run and then quickly search over
the DBSCAN* parameter space by extracting clustering results from our
pre-constructed tree.  This can save significant computational time when
searching across multiple cluster parameter settings on large amounts of data.

Alternatively, one could make use of the ``cluster_selection_epsilon`` as a
post processing step with any ``cluster_selection_method`` in order to
return a hybrid clustering of DBSCAN* and HDBSCAN*.  For more details on
this see :doc:`how_to_use_epsilon`.

In order to extract a DBSCAN* clustering from an HDBSCAN run we must first train
and HDBSCAN model on our data.

.. code:: python

    import hdbscan
    h_cluster = hdbscan.HDBSCAN(min_samples=5,match_reference_implementation=True).fit(X)

The ``min_cluster_size`` parameter is unimportant in this case in that it is
only used in the creation of our condensed tree which we won't be using here.
Now we choose a ``cut_distance`` which is just another name for the epsilon
threshold in DBSCAN and will be passed to our
:py:meth:`~hdbscan.hdbscan_.dbscan_clustering` method.

.. code:: python

    eps = 0.2
    labels = h_cluster.dbscan_clustering(cut_distance=eps, min_cluster_size=5)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels.astype(str));

.. image:: images/dbscan_from_hdbscan_clustering.png
    :align: center

It should be noted that a DBSCAN* clustering extracted from our HDBSCAN* tree will
not precisely match the clustering results from sklearn's DBSCAN implementation.
Our clustering results should better match DBSCAN* (which can be thought of as
DBSCAN without the border points).  As such when comparing the two results one
should expect them to mostly differ in the points that DBSCAN considers boarder
points.  We'll deal with
this by only looking at the comparison of our clustering results based on the points identified
by DBSCAN as core points.  We can see below that the differences between these two
clusterings mostly occur in the boundaries of the clusters.  This matches our
intuition of stability within the core points.

.. image:: images/dbscan_from_hdbscan_comparision.png
    :align: center

For a slightly more empirical comparison we we make use of the `adjusted rand score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html>`__
to compare the clustering of the core points between a DBSCAN cluster from sklearn and
a DBSCAN* clustering extracted from our HDBSCAN* object.

.. image:: images/dbscan_from_hdbscan_percentage_core.png
    :align: center

.. image:: images/dbscan_from_hdbscan_number_of_clusters.png
    :align: center

We see that for very small epsilon values our number of clusters tends to be quite
far apart, largely due to a large number of the points being considered boundary points
instead of core points.  As the epsilon value increases, more and more points are
considered core and the number of clusters generated by each algorithm converge.

Additionally, the adjusted rand score between the core points of both algorithm
stays consistently high (mostly 1.0) for our entire range of epsilon.  There may be
be some minor discrepancies between core point results largely due to implementation
details and optimizations with the code base.

Why might one just extract the DBSCAN* clustering results from a single HDBSCAN* run
instead of making use of sklearns DBSSCAN code?  The short answer is efficiency.
If you aren't sure what epsilon parameter to select for DBSCAN then you may have to
run the algorithm many times on your data set.  While those runs can be inexpensive for
very small epsilon values they can get quite expensive for large parameter values.

In this small benchmark case of 50,000 two dimensional data points we have broken even
after having only had to try two epsilon parameters from DBSCAN, or only a single
run with a large parameter selected.  This trend is only exacerbated for larger
data sets in higher dimensional spaces.  For more detailed scaling experiments see
`Accelearted Hierarchical Density Clustering <https://arxiv.org/abs/1705.07321>`__
by McInnes and Healy.

.. image:: images/dbscan_from_hdbscan_timing.png
    :align: center






