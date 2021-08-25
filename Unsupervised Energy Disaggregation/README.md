# Unsupervised Disaggregation by Sparse Signal Approximation and Dynamic Time Warping

This algorithm uses the UK-DALE NILM dataset from house 3.

This algorithm consists of 3 steps:
  1) Sparse signal approximation by Cholesky Orthogonal Matching Pursuit
  2) TimeSeries K-Means clustering
  3) Labeling by Dynamic Time Warping

UnsupervisedDisaggregationDemo.py contains the code for disaggregation. It is setup to create a synthetic dataset with a length of 2000 using the additional 5 appliance instance files included in the repository and then disaggregate from there. The commented-out parts of the code are to adapt the code to data of whatever length.

Everything in the HP-detection folder is not my work and is the source code from this paper http://pierrepinson.com/docs/Lerayetal2019-unsupnilm.pdf
The code is incomplete, but it served well for some reference so I've included it.
