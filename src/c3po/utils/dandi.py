"""Utilities for fetching NWB data from DANDI and training sorted-units C3PO models."""

from pathlib import Path
from dandi.consts import known_instances
from dandi.dandiapi import DandiAPIClient
from dandi.consts import known_instances
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import pynwb
from pathlib import Path
import h5py
import os
import pandas as pd

from c3po.utils.sorted import train_from_nwb
from c3po.utils.sorted import train_from_sorted_data as train_from_spike_data # backwards compatibility

# Helpers
def get_dandi_file(
    dandiset_id: str, dandi_path: str, dandi_instance: str = "dandi"
) -> tuple:
    """Stream an NWB file asset from a Dandiset, caching it to local disk.

    Args:
        dandiset_id (str): Dandiset identifier (e.g. "000123").
        dandi_path (str): Path to the asset within the dandiset.
        dandi_instance (str, optional): Dandi instance to query (e.g. "dandi" or
            "dandi-staging"). Defaults to "dandi".

    Returns:
        tuple: (io, nwbfile) where `io` is the open `pynwb.NWBHDF5IO` reader and
            `nwbfile` is the read `pynwb.NWBFile` object.
    """
    tmp_dir = os.environ.get("C3PO_CACHE_DIR", "dandi_cache")  # Local folder for cache
    Path(tmp_dir).mkdir(exist_ok=True)  # Create the cache folder if it doesn't exist

    # get the s3 url from Dandi
    with DandiAPIClient(
        dandi_instance=known_instances[dandi_instance],
    ) as client:
        asset = client.get_dandiset(dandiset_id).get_asset_by_path(dandi_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # stream the file from s3
    # first, create a virtual filesystem based on the http protocol
    fs = fsspec.filesystem("http")

    # create a cache to save downloaded data to disk (optional)
    fsspec_file = CachingFileSystem(
        fs=fs,
        cache_storage=tmp_dir,  # Local folder for cache
    )

    # Open and return the file
    fs_file = fsspec_file.open(s3_url, "rb")
    io = pynwb.NWBHDF5IO(file=h5py.File(fs_file))
    nwbfile = io.read()
    return io, nwbfile

def dandi_fetch_nwb_contents(entries:pd.DataFrame):
    """Fetch NWB contents from Dandi for the given entries.

    Args:
        entries (pd.DataFrame): DataFrame containing Dandi entries with columns
            'dandiset_id', 'dandi_path', and any '<object_type>_object_id' columns.

    Returns:
        list[dict]: List of dictionaries containing the fetched NWB objects.
    """
    results = []
    for row in entries.itertuples():
        io, nwbfile = get_dandi_file(
            dandiset_id= row.dandiset_id,
            dandi_path = row.dandi_path
        )
        object_dict = {}
        for col in row._fields:
            if col.endswith("_object_id"):
                object_id = getattr(row, col)
                data = nwbfile.objects[object_id]
                import pynwb
                if isinstance(data, pynwb.base.DynamicTable):
                    data = data.to_dataframe()
                object_dict[col.split("_object_id")[0]] = data

        results.append(object_dict)
    return results

# Training
def train_from_dandi(
    dandiset_id: str,
    dandi_path: str,
    dandi_instance: str = "dandi",
    dimensions: int = 16,
    output_dir: Path = Path("."),
    sample_length: int = 2000,
    train_args: dict = dict(),
    model_args: dict | Path | str = "sorted_default_v0",
):
    """Download an NWB asset from Dandi and train a sorted-units C3PO model on it.

    Args:
        dandiset_id (str): Dandiset identifier (e.g. "000123").
        dandi_path (str): Path to the NWB asset within the dandiset.
        dandi_instance (str): Dandi instance to query (e.g. "dandi" or "dandi-staging").
        dimensions (int): Latent/context dimensionality of the model.
        output_dir (Path): Directory to save the trained model and training outputs.
        sample_length (int, optional): Number of spike events per training sample.
            Defaults to 2000.
        train_args (dict, optional): Overrides for `DEFAULT_TRAIN_ARGS` passed to
            `train_model`. Defaults to an empty dict.
        model_args (dict | Path | str, optional): Model configuration, see
            `get_sorted_c3po_model`. Defaults to "sorted_default_v0".

    Returns:
        The result of `train_from_nwb` / `train_from_spike_data` (None; outputs are
        saved to `output_dir`).
    """
    io, nwbfile = get_dandi_file(dandiset_id, dandi_path, dandi_instance)
    return train_from_nwb(
        nwbfile,
        dimensions,
        output_dir,
        sample_length=sample_length,
        model_args=model_args,
        train_args=train_args,
    )
