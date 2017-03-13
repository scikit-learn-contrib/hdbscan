---
title: 'hdbscan: Hierarchical density based clustering'
tags:
  - clustering
  - unsupervised learning
  - machine learning
authors:
 - name: Leland McInnes
   orcid: 0000-0003-2143-6834
   affiliation: 1
 - name: John Healy
   affiliation: 1
 - name: Steve Astels
   affiliation: 2
affiliations:
 - name: Tutte Institute for Mathematics and Computing
   index: 1
 - name: Shopify
   index: 2
date: 26 February 2017
bibliography: paper.bib
---

# Summary

HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise 
[@campello2013density], [@campello2015hierarchical]. 
Performs DBSCAN over varying epsilon values and integrates the result to find a 
clustering that gives the best stability over epsilon. This allows HDBSCAN to 
find clusters of varying densities (unlike DBSCAN), and be more robust to parameter 
selection. The library also includes support for Robust Single Linkage clustering
[@chaudhuri2014consistent], [@chaudhuri2010rates],
GLOSH outlier detection [@campello2015hierarchical], and tools for visualizing 
and exploring cluster structures.
Finally support for prediction and soft clustering is also available.

-![Example clustering results.](hdbscan_clustering_result.png)
-![Hierarchical tree structure.](hdbscan_condensed_tree.png)

# References
  
