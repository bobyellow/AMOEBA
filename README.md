### AMOEBA
A faster version of AMOEBA algorithm (Aldstadt & Getis 2006) that detects irregular-shaped spatial clusters using a bottom-up approach.

AMOEBA (A Multidirectional Optimal Ecotope-Based Algorithm) follows a bottom-up strategy to identify irregular-shaped ecotopes of high or low values without having such false-positive error. It starts with one or more seed cell (spatial unit) to which neighboring cells are iteratively included until the maximum (or minimal) magnitude of the local spatial statistics, e.g. local G statistics (Getis and Ord 1992; Ord and Getis 1995) has been reached. 


**Key improvements** over the original AMOEBA:

  --Seed reduction & ranking: use head-tail breaks (Tao et al. 2020) to select only the top/bottom X % of values as seeds, sorted by standalone |G| so strongest hotspots/coldspots claim first.
  
  --Speed‐ups: use Python’s built-in sorted, in-loop random.sample for permutations, and optional multiprocessing to slash pure-Python overhead.
  
  --Robustness: filters NaNs, guards zero‐std, prunes overlaps in seed‐strength order, and (optionally) merges adjacent same-sign clusters.
  
  --Clear visualization: NS vs. small vs. major clusters with cold/hot color ramps, clean outside legend, equal‐aspect map, labeled scale bar & north arrow.


Example with Chicago gun violence data at the census tract level:
The map shows the clusters detected by the improved AMOEBA algorithm, where redish clusters are positive clusters (or hot spots) and blueish clusters are negative clusters (or cold spots). Non-significant units (NS) and some trivial clusters are colored in grey. 

![AMOEBA_Chicago_CT_GV_sig001_headtail01](https://github.com/user-attachments/assets/31b7c4cd-8b48-4a4b-8ceb-9fad9f3c8f1a)



### References:
Aldstadt, J., & Getis, A. (2006). Using AMOEBA to create a spatial weights matrix and identify spatial clusters. Geographical analysis, 38(4), 327-343.

Getis, A., Ord, J. (1992). “The Analysis of Spatial Association by Use of Distance Statistics.” Geographical Analysis 24 (3): 189–206.

Ord, J., and Getis, A. (1995). “Local Spatial Autocorrelation Statistics: Distributional Issues and an Application.” Geographical Analysis 27 (4): 286–306.

Tao, R., Gong, Z., Ma, Q., & Thill, J. C. (2020). Boosting computational effectiveness in big spatial flow data analysis with intelligent data reduction. ISPRS International Journal of Geo-Information, 9(5), 299.
