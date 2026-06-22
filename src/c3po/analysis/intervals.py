import numpy as np

def _is_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])


def _interval_list_contains_ind_sorted(interval_list, timestamps):
    ind = []
    for interval in interval_list:
        st = np.searchsorted(timestamps, interval[0], side="left")
        en = np.searchsorted(timestamps, interval[1], side="right")
        ind += list(range(st, en))
    ind = np.unique(ind)
    return np.asarray(ind)


def interval_list_contains_ind(interval_list, timestamps):
    """Find indices of list of timestamps contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    ind = []
    if _is_sorted(timestamps):
        return _interval_list_contains_ind_sorted(interval_list, timestamps)
    for interval in interval_list:
        ind += np.ravel(
            np.argwhere(
                np.logical_and(timestamps >= interval[0], timestamps <= interval[1])
            )
        ).tolist()
    ind = np.unique(ind)
    return np.asarray(ind)


def interval_list_contains(interval_list, timestamps):
    """Find timestamps that are contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    ind = interval_list_contains_ind(interval_list, timestamps)
    if len(ind) == 0:
        return np.array([])
    return timestamps[ind]


def interval_list_intersect(interval_list_1, interval_list_2):
    """Find the intersection of two interval lists.

    Parameters
    ----------
    interval_list_1 : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    interval_list_2 : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    """
    intersected_intervals = []
    for interval_1 in interval_list_1:
        for interval_2 in interval_list_2:
            start = max(interval_1[0], interval_2[0])
            end = min(interval_1[1], interval_2[1])
            if start < end:
                intersected_intervals.append((start, end))
    intersect = np.array(intersected_intervals)
    # Handle the empty case explicitly to ensure shape (0, 2)
    if intersect.size == 0:
        return np.empty((0, 2), dtype=intersect.dtype)
    # If there is exactly one interval, ensure shape (1, 2)
    if intersect.ndim == 1 and intersect.shape[0] == 2:
        intersect = intersect[None, :]
    return intersect


def interval_list_complement(intervals1, intervals2):
    """
    Finds intervals in `intervals1` that are not covered by any interval in `intervals2`.

    Parameters
    ----------
    intervals1 : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    intervals2 : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    """
    result = []
    for start1, end1 in intervals1:
        subtracted = [(start1, end1)]
        for start2, end2 in intervals2:
            new_subtracted = []
            for s, e in subtracted:
                if start2 <= s and e <= end2:
                    continue
                if e <= start2 or end2 <= s:
                    new_subtracted.append((s, e))
                    continue
                if start2 > s:
                    new_subtracted.append((s, start2))
                if end2 < e:
                    new_subtracted.append((end2, e))
            subtracted = new_subtracted
        result.extend(subtracted)
    return np.array(result)