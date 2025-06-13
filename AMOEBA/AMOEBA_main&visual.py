import core.shapefile
import numpy as np
import pandas as pd
import os
from scipy import stats
from core.getNeighbors import getNeighborsAreaContiguity
#from core.AMOEBA import execAMOEBA #original AMOEBA
#an improved faster version of AMOEBA using head-tail break for seeds selection:
from core.AMOEBA_headtail_fast import execAMOEBA_headtail_fast
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Load shapefile data
SHAPEFILE_PATH = "input/Chicago_census_tract.shp"
##sf = core.shapefile.Reader("input/US_County_Lower48.shp")
sf = core.shapefile.Reader(SHAPEFILE_PATH)
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

# Build univariate value dictionary for AMOEBA
y = dict(zip(df["O"], df["F"]))

# Run AMOEBA with desired p-value threshold
significance_level = 0.01
#outputStr = execAMOEBA(y, neighbors, significance=significance_level)
outputStr = execAMOEBA_headtail_fast(y, neighbors, significance=significance_level,mc_reps=500, headtail_pct=0.05)

# Save output to file
#output_path = "result/AMOEBA_County_Obesity_001sig_headtail_order_01.txt"
output_path = "result/AMOEBA_Chicago_CT_GV_sig001_headtail005.txt"

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write(outputStr)

print(f"Processing complete. Results written to {output_path}")


# --------------Visualize results in map---------------------------------------
# 0) CONFIG — adjust as needed
# -----------------------------------------------------------------------------
#SHAPEFILE_PATH = "input/data_prep_tract_final.shp"
RESULT_PATH    = output_path
OUTPUT_MAP     = output_path.replace('txt', 'jpg')
MIN_PCT        = 0.01   # 1% threshold for grouping “small” clusters of which unit number is less than 1% of total

# -----------------------------------------------------------------------------
# 1) LOAD SHAPEFILE
# -----------------------------------------------------------------------------
sf      = core.shapefile.Reader(SHAPEFILE_PATH)
shapes  = sf.shapes()
n_units = len(shapes)

# -----------------------------------------------------------------------------
# 2) LOAD RESULTS
# -----------------------------------------------------------------------------
res = pd.read_csv(RESULT_PATH, sep=",")
res.columns = [c.strip() for c in res.columns]
if "Area" not in res or "AMOEBA" not in res:
    raise KeyError("Result file must contain 'Area' and 'AMOEBA' columns")
res["Area"]   = res["Area"].astype(int)
res["AMOEBA"] = res["AMOEBA"].astype(int)
results_dict = dict(zip(res["Area"], res["AMOEBA"]))

# -----------------------------------------------------------------------------
# 3) BUILD PATCHES & RAW CODES
# -----------------------------------------------------------------------------
patches   = []
raw_codes = []

for idx, shape in enumerate(shapes, start=1):
    code = results_dict.get(idx, 0)
    pts  = shape.points
    parts = list(shape.parts) + [len(pts)]
    for start, end in zip(shape.parts, parts[1:]):
        patches.append(Polygon(pts[start:end], closed=True))
        raw_codes.append(code)

# -----------------------------------------------------------------------------
# 4) CLASSIFY INTO NS / small / others
# -----------------------------------------------------------------------------
from collections import Counter
counter     = Counter(raw_codes)
min_size    = max(1, int(n_units * MIN_PCT))
small_codes = {c for c, cnt in counter.items() if c != 0 and cnt < min_size}

# build display codes
disp_codes = []
for c in raw_codes:
    if c == 0:
        disp_codes.append("NS")
    elif c in small_codes:
        disp_codes.append("small")
    else:
        disp_codes.append(str(c))

disp_counts = Counter(disp_codes)

# -----------------------------------------------------------------------------
# 5) ASSIGN COLORS
# -----------------------------------------------------------------------------
# greys for NS & small
color_map = {
    "NS":    "#4d4d4d",
    "small": "#cccccc",
}

# separate negative and positive codes
others = sorted([d for d in disp_counts if d not in ("NS","small")], key=int)
neg_codes = [c for c in others if int(c) < 0]
pos_codes = [c for c in others if int(c) > 0]

# cold blues for negatives (darkest for largest |c|)
cold = plt.get_cmap("Blues")
n_neg = len(neg_codes)
for i, code in enumerate(sorted(neg_codes, key=lambda x: int(x))):
    # reversed so that more negative = darker
    color_map[code] = cold(1 - i/(n_neg-1 if n_neg>1 else 1)*0.7)

# warm reds for positives (light → dark)
warm = plt.get_cmap("Reds")
n_pos = len(pos_codes)
for i, code in enumerate(sorted(pos_codes, key=lambda x: int(x))):
    color_map[code] = warm(0.3 + i/(n_pos-1 if n_pos>1 else 1)*0.7)

# now build facecolor list
facecolors = [color_map[c] for c in disp_codes]

# -----------------------------------------------------------------------------
# 6) PLOT
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,8))
pcol = PatchCollection(
    patches,
    facecolor=facecolors,
    edgecolor="black",
    linewidths=0.2,
)
ax.add_collection(pcol)
ax.set_aspect("equal", "box")
ax.autoscale()
ax.set_axis_off()

# -----------------------------------------------------------------------------
# 7) CUSTOM LEGEND (left of map)
# -----------------------------------------------------------------------------
handles = []

# then positive clusters
for code in pos_codes:
    handles.append(mpatches.Patch(color=color_map[code],
                                  label=f"{code} ({disp_counts[code]} units)"))
    # then negative clusters
for code in neg_codes:
    handles.append(mpatches.Patch(color=color_map[code],
                                  label=f"{code} ({disp_counts[code]} units)"))
# NS first
handles.append(mpatches.Patch(color=color_map["NS"],
                              label=f"NS ({disp_counts['NS']} units)"))
# small clusters next
handles.append(mpatches.Patch(color=color_map["small"],
                              label=f"small clusters (<{int(100*MIN_PCT)}%) ({disp_counts['small']} units)"))

leg = ax.legend(
    handles=handles,
    title="AMOEBA Cluster",
    loc="center left",
    bbox_to_anchor=(-0.3, 0.3),
    fontsize="medium",
    title_fontsize="large",
    frameon=False  # no border
)

# -----------------------------------------------------------------------------
# 8) SAVE JPEG
# -----------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_MAP), exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=300, format="jpg")
print(f"Map saved to {OUTPUT_MAP}")
