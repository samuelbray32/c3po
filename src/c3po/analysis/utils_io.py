from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def save_params_hdf5(params: Mapping[str, Any], path: str | Path) -> None:
    """Save nested model parameters to an HDF5 file.

    Parameters
    ----------
    params : Mapping[str, Any]
        Nested parameter tree. Leaves should be array-like objects, such as
        numpy arrays or JAX arrays. Example leaf shapes include
        ``(n_input, n_hidden)`` for dense kernels and ``(n_hidden,)`` for biases.
    path : str | Path
        Output HDF5 file path.

    """
    path = Path(path)

    with h5py.File(path, "w") as h5_file:
        _write_group(h5_file, params)


def _write_group(group: h5py.Group, tree: Mapping[str, Any]) -> None:
    """Recursively write a nested mapping to an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        Current HDF5 group.
    tree : Mapping[str, Any]
        Nested mapping whose leaves are array-like objects.

    """
    for key, value in tree.items():
        key = str(key)

        if isinstance(value, Mapping):
            subgroup = group.create_group(key)
            _write_group(subgroup, value)
        else:
            array = np.asarray(value)
            dataset = group.create_dataset(
                key,
                data=array,
                compression="gzip",
            )
            dataset.attrs["dtype"] = str(array.dtype)
            dataset.attrs["shape"] = array.shape


def load_params_hdf5(path: str | Path) -> dict[str, Any]:
    """Load nested model parameters from an HDF5 file.

    Parameters
    ----------
    path : str | Path
        Input HDF5 file path.

    Returns
    -------
    dict[str, Any]
        Nested parameter tree. Leaves are numpy arrays with their original
        shapes and dtypes.

    """
    path = Path(path)

    with h5py.File(path, "r") as h5_file:
        return _read_group(h5_file)


def _read_group(group: h5py.Group) -> dict[str, Any]:
    """Recursively read an HDF5 group into a nested dictionary.

    Parameters
    ----------
    group : h5py.Group
        Current HDF5 group.

    Returns
    -------
    dict[str, Any]
        Nested dictionary where leaves are numpy arrays.

    """
    output: dict[str, Any] = {}

    for key, value in group.items():
        if isinstance(value, h5py.Group):
            output[key] = _read_group(value)
        elif isinstance(value, h5py.Dataset):
            output[key] = np.asarray(value)
        else:
            raise TypeError(f"Unsupported HDF5 object at key {key}: {type(value)}")

    return output


# --------------
# argument params as json files
from pathlib import Path
from typing import Any

import json
import numpy as np


def make_json_serializable(value: Any) -> Any:
    """Convert Python/NumPy objects into JSON-serializable objects.

    Parameters
    ----------
    value : Any
        Object to convert. May be a nested dictionary, list, tuple, NumPy scalar,
        NumPy array, or primitive Python type.

    Returns
    -------
    Any
        JSON-serializable version of ``value``.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def write_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a configuration dictionary to disk as JSON.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    path : str | Path
        Output path.
    """
    path = Path(path)
    serializable_config = make_json_serializable(config)

    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable_config, file, indent=2, sort_keys=True)


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a configuration dictionary from a JSON file.

    Parameters
    ----------
    path : str | Path
        Input path.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)