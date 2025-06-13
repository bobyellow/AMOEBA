import time
import numpy as np
import random
from core.spatstats import calculateGetisG

__all__ = ['execAMOEBA_headtail_fast']


def execAMOEBA_headtail_fast(y, w, significance=0.01, mc_reps=500, headtail_pct=0.1):
    """
    Head‐Tail seeded AMOEBA, with seeds ranked by their individual local‐G strength.

    Steps:
      1. Pick top & bottom headtail_pct as seeds.
      2. Compute each seed’s standalone G = calculateGetisG([seed], …).
      3. Sort those seeds by descending |G|.
      4. Grow clusters in that order and prune overlaps as usual.

    Returns a CSV‐style string: "Area, AMOEBA, Value"
    """

    # --- 0) Filter out NaNs ---
    all_keys   = list(y.keys())
    valid_keys = [k for k in all_keys if not np.isnan(y.get(k, np.nan))]
    if not valid_keys:
        raise ValueError("No valid (non-NaN) values in y")
    N = len(valid_keys)

    # --- 1) Basic stats ---
    vals = np.array([y[k] for k in valid_keys], dtype=float)
    mean = np.nanmean(vals)
    std  = np.nanstd(vals)
    if std == 0:
        raise ValueError("Zero standard deviation")

    # --- 2) Head–tail seed selection ---
    sorted_by_value = sorted(valid_keys, key=y.__getitem__)
    n_seed = max(1, int(headtail_pct * N))
    headseeds = sorted_by_value[-n_seed:]
    tailseeds = sorted_by_value[:n_seed]
    seeds = headseeds + tailseeds

    # --- 3) Compute each seed’s standalone local G (“potential”) ---
    potentials = {}
    for s in seeds:
        Gs = calculateGetisG([s], mean, std, y, N)
        potentials[s] = 0.0 if np.isnan(Gs) else Gs

    # --- 4) Sort seeds by descending |potential| ---
    seeds_sorted = sorted(seeds, key=lambda s: abs(potentials[s]), reverse=True)

    # --- 5) Prepare neighbor lookup ---
    neigh = {k: [nb for nb in w.get(k, []) if nb in valid_keys] for k in valid_keys}

    # --- 6) AMOEBA cluster‐growing in seed order ---
    output = {k: 0 for k in all_keys}
    pos_id, neg_id = 1, -1
    start = time.time()

    for seed in seeds_sorted:
        # Skip if seed value was NaN
        if seed not in valid_keys:
            continue

        # Skip if this seed’s potential was NaN (no meaningful G)
        pot = potentials[seed]
        if pot == 0.0:
            continue

        # Grow initial cluster
        cluster = [seed]
        Gcurr   = pot

        # Iteratively expand
        while True:
            previous = Gcurr
            # frontier = neighbors of current cluster minus itself
            frontier = {nb for h in cluster for nb in neigh.get(h, [])} - set(cluster)
            if not frontier:
                break

            # sort frontier to try best neighbor first
            if Gcurr >= 0:
                frontier = sorted(frontier, key=y.__getitem__, reverse=True)
            else:
                frontier = sorted(frontier, key=y.__getitem__)

            improved = False
            for nb in frontier:
                cand = cluster + [nb]
                Gnew = calculateGetisG(cand, mean, std, y, N)
                if np.isnan(Gnew):
                    continue
                if (Gcurr >= 0 and Gnew > Gcurr) or (Gcurr < 0 and Gnew < Gcurr):
                    cluster = cand
                    Gcurr   = Gnew
                    improved = True
                    break

            if not improved or Gcurr == previous:
                break

        # Monte Carlo significance
        count = 0
        for _ in range(mc_reps):
            perm = random.sample(valid_keys, N)
            perm_map = {valid_keys[i]: perm[i] for i in range(N)}
            perm_cluster = [perm_map[h] for h in cluster]
            Gr = calculateGetisG(perm_cluster, mean, std, y, N)
            if np.isnan(Gr):
                continue
            if (Gcurr >= 0 and Gr > Gcurr) or (Gcurr < 0 and Gr < Gcurr):
                count += 1

        pval = count / mc_reps
        if pval <= significance:
            cid = pos_id if Gcurr > 0 else neg_id
            if Gcurr > 0:
                pos_id += 1
            else:
                neg_id -= 1
            # assign cluster ID only if not already taken
            for h in cluster:
                if output[h] == 0:
                    output[h] = cid

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f}s")

    # --- 7) Build CSV‐style output ---
    lines = ["Area, AMOEBA, Value"]
    for idx, k in enumerate(all_keys, start=1):
        val = y.get(k, np.nan)
        lines.append(f"{idx}, {output[k]}, {val}")
    return "\n".join(lines)
