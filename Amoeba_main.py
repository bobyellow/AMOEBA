import core.shapefile
import numpy as np
import pandas as pd
import os
from scipy import stats
from core.getNeighbors import getNeighborsAreaContiguity
#from core.AMOEBA import execAMOEBA #original AMOEBA
#an improved faster version of AMOEBA using head-tail break for seeds selection:
from core.AMOEBA_headtail_fast import execAMOEBA_headtail_fast

# Load shapefile data
##sf = core.shapefile.Reader("input/US_County_Lower48.shp")
sf = core.shapefile.Reader("input/Chicago_census_tract.shp")
shapes = sf.shapes()

# Prepare AREAS input for Queen's and Rook's contiguity
AREAS = [[shape.points] for shape in shapes]

# Calculate neighbors using Queen's contiguity
Wqueen, _ = getNeighborsAreaContiguity(AREAS)
neighbors = Wqueen

# Load and extract raw data from CSV
# Assumes whitespace-separated file with no header: columns O (ID) and F (value)

##df = pd.read_csv(
##    "input/County_48_Obesity.csv",
##    sep=r"\s+", header=None, names=["O", "F"]
##)
df = pd.read_csv(
    "input/Chicago_CT_GV.csv",
    sep=r"\s+", header=None, names=["O", "F"]
)
##df = pd.read_csv(
##    "input/Baidu_short_toWeihai_170401_07_OnlyO_from0_Log_ready.txt",
##    sep=r"\s+", header=None, names=["O", "F"]
##)
# Build univariate value dictionary for AMOEBA
y = dict(zip(df["O"], df["F"]))

# Run AMOEBA with desired p-value threshold
significance_level = 0.01
#outputStr = execAMOEBA(y, neighbors, significance=significance_level)
outputStr = execAMOEBA_headtail_fast(y, neighbors, significance=significance_level,mc_reps=500, headtail_pct=0.3)

# Save output to file
#output_path = "result/AMOEBA_County_Obesity_001sig_headtail_order_01.txt"
output_path = "result/AMOEBA_Chicago_CT_GV_001sig_headtail_order_03.txt"
#output_path = "result/AMOEBA_Weihai_001sig_headtail_01.txt"
# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write(outputStr)

print(f"Processing complete. Results written to {output_path}")
