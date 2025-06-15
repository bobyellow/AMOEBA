import time
import numpy as np
import random
from core.spatstats import calculateGetisG

__all__ = ['execAMOEBA_headtail_fast']

def execAMOEBA_headtail_fast(y, w, significance=0.01, mc_reps=500, headtail_pct=0.1):
    """
    Head-Tail seeded AMOEBA (faster version), skipping missing values.

    Only the top and bottom `headtail_pct` fraction of non-NaN units are used as seeds.

    Parameters:
    - y: dict of area_key -> value (may contain np.nan)
    - w: dict of area_key -> list of neighbor area_keys
    - significance: float, p-value threshold (default 0.01)
    - mc_reps: int, number of Monte Carlo permutations (default 500)
    - headtail_pct: float, fraction of units to use as seeds at each tail (default 0.1)

    Returns:
    - A string with area-level cluster assignments and values
    """
    start = time.time()
    # Separate valid and missing-value keys
    all_keys = list(y.keys())
    valid_keys = [k for k in all_keys if not np.isnan(y.get(k, np.nan))]
    N = len(valid_keys)
    if N == 0:
        raise ValueError("No valid (non-NaN) values found in y.")

    # Compute overall stats ignoring NaNs
    values = np.array([y[k] for k in valid_keys], dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0:
        raise ValueError("Standard deviation of valid values is zero.")

    # Precompute neighbor lists for valid keys
    neigh = {k: [nb for nb in w.get(k, []) if nb in valid_keys] for k in valid_keys}

    # Head-tail seed selection among valid keys
    sorted_keys = sorted(valid_keys, key=y.__getitem__)
    n_seed = max(1, int(headtail_pct * N))
    seeds = sorted_keys[:n_seed] + sorted_keys[-n_seed:]

    # Initialize output: 0 for unassigned, NaN keys left as 0
    output = {k: 0 for k in all_keys}
    pos_id, neg_id = 1, -1

    for seed in seeds:
        # Grow cluster from seed
        cluster = [seed]
        Gcurr = calculateGetisG(cluster, mean, std, y, N)
        if np.isnan(Gcurr):
            continue
        improved = True
        while improved:
            improved = False
            frontier = set(nb for h in cluster for nb in neigh.get(h, [])) - set(cluster)
            if not frontier:
                break
            # Sort frontier based on current cluster sign
            if Gcurr >= 0:
                sorted_fr = sorted(frontier, key=y.__getitem__, reverse=True)
            else:
                sorted_fr = sorted(frontier, key=y.__getitem__)
            for nb in sorted_fr:
                cand = cluster + [nb]
                Gnew = calculateGetisG(cand, mean, std, y, N)
                if np.isnan(Gnew):
                    continue
                if (Gcurr >= 0 and Gnew > Gcurr) or (Gcurr < 0 and Gnew < Gcurr):
                    cluster = cand
                    Gcurr = Gnew
                    improved = True
                    break

        # Skip singleton clusters
        if len(cluster) < 2:
            continue

        # Monte Carlo significance test
        better = 0
        for _ in range(mc_reps):
            perm = random.sample(valid_keys, N)
            perm_map = {valid_keys[i]: perm[i] for i in range(N)}
            perm_cluster = [perm_map[h] for h in cluster]
            Gr = calculateGetisG(perm_cluster, mean, std, y, N)
            if np.isnan(Gr):
                continue
            if (Gcurr >= 0 and Gr > Gcurr) or (Gcurr < 0 and Gr < Gcurr):
                better += 1
        p = better / float(mc_reps)
        if p <= significance:
            cid = pos_id if Gcurr > 0 else neg_id
            if Gcurr > 0:
                pos_id += 1
            else:
                neg_id -= 1
            for h in cluster:
                if output[h] == 0:
                    output[h] = cid

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f}s")

    # Build result output
    lines = ["Area, AMOEBA, Value"]
    for idx, k in enumerate(all_keys):
        val = y.get(k, np.nan)
        lines.append(f"{idx+1}, {output[k]}, {val}")
    return "\n".join(lines)
