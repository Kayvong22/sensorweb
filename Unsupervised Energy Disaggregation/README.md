# Unsupervised Disaggregation by Sparse Signal Approximation and Dynamic Time Warping

This algorithm uses the UK-DALE NILM dataset from house 3.

This algorithm consists of 3 steps:
  1) Sparse signal approximation by Cholesky Orthogonal Matching Pursuit
  2) TimeSeries K-Means clustering
  3) Labeling by Dynamic Time Warping

UnsupervisedDisaggregationDemo.py contains the code for disaggregation. It is setup to create a synthetic dataset with a length of 2000 using the additional 5 appliance instance files included in the repository and then disaggregate from there. The commented-out parts of the code are to adapt the code to data of whatever length.
