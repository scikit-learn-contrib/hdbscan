Frequently Asked Questions
==========================

Here we attempt to address some common questions, directing the user to some
helpful answers.

Q: Most of data is classified as noise; why?
--------------------------------------------

The amount of data classified as noise is controlled by the ``min_samples``
parameter. By default, if not otherwise set, this value is set to the same
value as ``min_cluster_size``. You can set it independently if you wish by
specifying it separately. The lower the value, the less noise you'll get, but
 there are limits, and it is possible that you simply have noisy data. See
:ref:`_min_samples_label` for more details.

Q: I mostly just get one large cluster; I want smaller clusters.
----------------------------------------------------------------

If you are getting a single large cluster and a few small outlying clusters
that means your data is essentially a large glob with some small outlying
clusters -- there may be structure to the glob, but compared to how well
separated those other small clusters are, it doesn't really show up. You may,
 however, want to get at that more fine grained structure. You can do that,
and what you are looking for is leaf clustering :ref:`_leaf_cluster_label` .