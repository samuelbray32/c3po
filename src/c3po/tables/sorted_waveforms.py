import datajoint as dj

from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.common.custom_nwbfile import AnalysisNwbfile

from spyglass.decoding.v1.waveform_features import (
    WaveformFeaturesParams,
    UnitWaveformFeatures,
)

from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.decoding.v1.c3po import MarksGroup, Model

import numpy as np
import pandas as pd

schema = dj.schema("sb_sorted_waveforms")


@schema
class SortedWaveformSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id="merge_id")
    -> WaveformFeaturesParams
    """


@schema
class SortedWaveformFeatures(SpyglassMixin, dj.Computed):
    definition = """
    -> SortedWaveformSelection
    ---
    -> AnalysisNwbfile
    waveform_object_id: varchar(40)
    """

    _parallel_make = True

    def fetch1_dataframe(self, key=dict()):
        if not len(self & key) == 1:
            raise ValueError(
                f"Expected one entry for key {key}, found {len(self & key)}"
            )
        return (self & key).fetch_nwb()[0]["waveform"]

    def make(self, key):
        #  get the list of feature parameters
        params = (WaveformFeaturesParams & key).fetch1("params")

        #  check that the feature type is supported
        if not WaveformFeaturesParams.check_supported_waveform_features(
            params["waveform_features_params"]
        ):
            raise NotImplementedError(
                f"Features {set(params['waveform_features_params'])} are "
                + "not supported"
            )

        merge_key = {"merge_id": key["spikesorting_merge_id"]}
        # load unit_df
        spike_times = (SpikeSortingOutput() & merge_key).fetch_nwb()[0]["units"]

        # pull out full waveforms
        waveform_extractor = UnitWaveformFeatures._fetch_waveform(
            merge_key, params["waveform_extraction_params"]
        )

        source_key = SpikeSortingOutput().merge_get_parent(merge_key).fetch1()
        sorter = source_key["sorter"]
        nwb_file_name = source_key["nwb_file_name"]
        analysis_nwb_key = "units"

        waveform_features = {}
        for feature, feature_params in params["waveform_features_params"].items():
            waveform_features[feature] = (
                UnitWaveformFeatures._compute_waveform_features(
                    waveform_extractor,
                    feature,
                    feature_params,
                    sorter,
                )
            )

        unit_ids = [
            int(i)
            for i in waveform_extractor.sorting.get_unit_ids()
            if i in spike_times.index
        ]
        # create new analysis nwb file
        analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
        import pandas as pd

        df = {}
        for unit_id in unit_ids:
            df[unit_id] = {}
            for metric, metric_dict in waveform_features.items():
                df[unit_id][metric] = (
                    metric_dict[unit_id] if unit_id in metric_dict else []
                )
            df[unit_id]["spike_times"] = spike_times.spike_times.loc[unit_id]
        df = pd.DataFrame.from_dict(df, orient="index")
        waveform_object_id = AnalysisNwbfile().add_nwb_object(analysis_nwb_file, df)

        AnalysisNwbfile().add(nwb_file_name, analysis_nwb_file)
        new_key = {
            **key,
            "analysis_file_name": analysis_nwb_file,
            "waveform_object_id": waveform_object_id,
        }
        self.insert1(new_key)


@schema
class EmbeddedSortedWaveformSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortedWaveformFeatures
    -> Model
    checkpoint = -1 : int
    """


@schema
class EmbeddedSortedWaveform(SpyglassMixin, dj.Computed):
    definition = """
    -> EmbeddedSortedWaveformSelection
    ---
    -> AnalysisNwbfile
    embedded_waveform_object_id: varchar(40)
    """

    def make(self, key):
        # get the shank id for the ss_group
        # sort group for the sorted spikes
        target_group = (
            SpikeSortingOutput().get_sort_group_info(
                {"merge_id": key["spikesorting_merge_id"]}
            )
        ).fetch1("sort_group_id")
        # find the clusterless merge_id with the same sort group as the target group
        model_key = (Model & key).fetch1("KEY")
        query = MarksGroup().WaveformFeatures & model_key
        merge_ids = query.fetch("spikesorting_merge_id")
        match_id = None
        for merge_id in merge_ids:
            sort_group = (
                SpikeSortingOutput().get_sort_group_info({"merge_id": merge_id})
            ).fetch1("sort_group_id")
            if sort_group == target_group:
                print(f"Found matching merge_id: {merge_id}")
                match_id = merge_id
                break
        if match_id is None:
            raise ValueError("No matching merge_id found for the target group.")

        # determine the shank index of the matching merge_id used in the trained model
        load_order = (UnitWaveformFeatures & query).fetch(
            "spikesorting_merge_id",
        )
        shank_index = None
        for i, merge_id in enumerate(load_order):
            if merge_id == match_id:
                shank_index = i
                break
        if shank_index is None:
            raise ValueError("Matching merge_id not found in load order.")

        # trained scaling factor for this shank
        clusterless_wf = (
            UnitWaveformFeatures()
            & query
            & {
                "spikesorting_merge_id": match_id,
            }
        ).fetch_data()[1][0]
        norm_scale = np.max(np.abs(clusterless_wf), axis=0)

        # fetch the wavform features for the sorted data
        ss_wf_df = (SortedWaveformFeatures & key).fetch1_dataframe()

        # load the trained model
        q = Model() & key
        analysis = q.fetch_c3po_analysis(
            checkpoint=key["checkpoint"] if key["checkpoint"] != -1 else None
        )

        # embed the sorted waveforms using the trained model
        results = dict()
        for row in ss_wf_df.itertuples():
            unit_id = row.index
            waveforms = row.amplitude
            # normalize the waveforms using the scaling factor from training
            waveforms = waveforms / norm_scale
            waveforms = waveforms - np.mean(waveforms, axis=1)[:, None]
            marks = np.zeros((len(waveforms), waveforms[0].size + 1))
            marks[:, : waveforms.shape[1]] = waveforms
            marks[:, -1] = shank_index
            dt = np.ones((len(waveforms)))
            z, _, __ = analysis.embed_data(
                marks[None, ...],
                dt[None, ...],
                first_mark_time=0.0,
                chunk_size=5000,
                chunk_padding=0,
                delta_t_units="ms",
                store_data=False,
                chunk_data=True,
            )
            results[unit_id] = z

        # build df of results
        df = ss_wf_df.copy()
        df["z"] = results.values()
        df.pop("amplitude")

        # create new analysis nwb file
        with self.analysis_table.build(key["nwb_file_name"]) as builder:
            # store results in file
            object_id = builder.add_nwb_object(df, "embedded_waveform")
            analysis_file_name = builder.analysis_file_name

        new_key = {
            **key,
            "analysis_file_name": analysis_file_name,
            "embedded_waveform_object_id": object_id,
        }
        self.insert1(new_key)

    def fetch1_dataframe(self, key=dict()):
        if not len(self & key) == 1:
            raise ValueError(
                f"Expected one entry for key {key}, found {len(self & key)}"
            )
        return (self & key).fetch_nwb()[0]["embedded_waveform"]

    def fetch_dataframe(self):
        results = [nwb["embedded_waveform"] for nwb in self.fetch_nwb()]
        return pd.concat(results, axis=0)

