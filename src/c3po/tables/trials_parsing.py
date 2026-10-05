from spyglass.common import DIOEvents, Session, IntervalList
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_mixin import SpyglassMixin

import pandas as pd
import numpy as np
import datajoint as dj

schema = dj.schema("sambray_trial_parsing")


@schema
class DIOGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    dio_group_name: varchar(32)
    """

    class DIOGroupMember(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> DIOEvents
        description: varchar(64)
        """


@schema
class TrialsSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DIOGroup
    -> IntervalList
    task_name: varchar(32)
    """


@schema
class Trials(SpyglassMixin, dj.Computed):
    definition = """
    -> TrialsSelection
    ---
    -> AnalysisNwbfile
    trials_object_id: varchar(64)  # Object ID of the trials table in the NWB file
    """

    def make(self, key):
        raise NotImplementedError(
            "Trials tables in C3PO are a stub for loading and import only"
        )
