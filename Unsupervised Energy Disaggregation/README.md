# Unsupervised Energy Disaggregation:
# From Sparse Signal Approximation to Community Detection

This algorithm for now uses the UK-DALE NILM dataset from house 3.

This algorithm consists of 3 steps:
  1) Sparse signal approximation
  2) TimeSeriesKMeans clustering
  3) Labeling by Dynamic Time Warping

Everything in the folder HP-detection is source code from the author. The file DemoNew.py is my most updated version of my code. It will be updated again shortly.

syntheticdata.py is an example of how to create a synthetic dataset of time series length 2000 and then separate and classify type 1 (just on/off) appliances. 
