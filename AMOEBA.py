import time as tm
import numpy as np
from core.spatstats import calculateGetisG

__all__ = ['execAMOEBA']

def quickSort2(keys, y):
    """
    Recursively sort keys by their corresponding values in y using quicksort.
    """
    if len(keys) <= 1:
        return keys
    pivot_key = keys[-1]
    pivot_val = y[pivot_key]
    rest = keys[:-1]
    less = [k for k in rest if y[k] <= pivot_val]
    greater = [k for k in rest if y[k] > pivot_val]
    return quickSort2(less, y) + [pivot_key] + quickSort2(greater, y)


def neighborSort(dictionary, discardList):
    """
    Return keys of dictionary sorted by their values, excluding discardList.
    """
    sorted_keys = quickSort2(list(dictionary.keys()), dictionary)
    return [k for k in sorted_keys if k not in discardList]


def execAMOEBA(y, w, significance=0.01):
    """
    AMOEBA: A Multidirectional Optimum Ecotope-Based Algorithm

    Parameters:
    - y: dict of area_key -> value
    - w: dict of area_key -> list of neighbor area_keys
    - significance: float, p-value threshold (default 0.01)

    Returns:
    - A string with area-level cluster assignments and values
    """
    start = tm.time()
    #print("Running computationally efficient AMOEBA (Duque et al., 2010)")
    #print("Number of areas:", len(y))

    areaKeys = list(y.keys())
    values = list(y.values())
    dataMean = np.mean(np.asarray(values, dtype=float))
    dataStd = np.std(np.asarray(values, dtype=float))
    dataLength = len(y)

    generatedClusters = {}
    clusterGValues = {}
    clusterGValuesAbs = {}

    print("Starting iterative process")
    for s in areaKeys:
        neighbors = w.get(s, [])
        itAreaList = [s]
        currentG = calculateGetisG(itAreaList, dataMean, dataStd, y, dataLength)
        previousG = None

        # grow cluster until G stabilizes
        while currentG != previousG:
            previousG = currentG
            sortedNeighbors = quickSort2(neighbors, y)
            # expand cluster
            if currentG <= 0:
                # cold-spot: include lowest-value neighbors first
                for a in range(len(sortedNeighbors)):
                    group = itAreaList + sortedNeighbors[:a+1]
                    newG = calculateGetisG(group, dataMean, dataStd, y, dataLength)
                    if newG < currentG:
                        currentG = newG
                        itAreaList = group
            else:
                # hot-spot: include highest-value neighbors first
                for a in range(len(sortedNeighbors)):
                    group = itAreaList + sortedNeighbors[-(a+1):]
                    newG = calculateGetisG(group, dataMean, dataStd, y, dataLength)
                    if newG > currentG:
                        currentG = newG
                        itAreaList = group
            # update frontier neighbors
            neighbors = []
            for area in itAreaList:
                for nb in w.get(area, []):
                    if nb not in itAreaList and nb not in neighbors:
                        neighbors.append(nb)

        generatedClusters[s] = itAreaList
        clusterGValues[s] = currentG
        clusterGValuesAbs[s] = abs(currentG)

    # prioritize clusters by strength
    prioritary = list(neighborSort(clusterGValuesAbs, []))[::-1]

    # Monte Carlo permutations
    print("Testing cluster significance")
    areaRange = list(range(dataLength))
    randomKeyList = []
    for _ in range(1000):
        perm = np.random.permutation(areaRange)
        randomKeyList.append([areaKeys[i] for i in perm])

    output = {k: 0 for k in areaKeys}
    clusterIndex = {areaKeys[i]: i for i in range(dataLength)}
    posId, negId = 1, -1

    for x in prioritary:
        cluster = generatedClusters[x]
        if any(output[h] != 0 for h in cluster):
            continue
        better = 0
        for j in range(1000):
            permKeys = randomKeyList[j]
            permCluster = [permKeys[clusterIndex[h]] for h in cluster]
            randomG = calculateGetisG(permCluster, dataMean, dataStd, y, dataLength)
            if (clusterGValues[x] >= 0 and randomG > clusterGValues[x]) or \
               (clusterGValues[x] < 0 and randomG < clusterGValues[x]):
                better += 1
        pValue = better / 1000.0
        if pValue <= significance:
            cid = posId if clusterGValues[x] > 0 else negId
            if clusterGValues[x] > 0:
                posId += 1
            else:
                negId -= 1
            for h in cluster:
                output[h] = cid

    elapsed = tm.time() - start
    print(f"Elapsed time: {elapsed:.2f}s")

    # build result output
    lines = ["Area, AMOEBA, Value"]
    for i in areaKeys:
        lines.append(f"{i+1}, {output[i]}, {y[i]}")
    return "\n".join(lines)
