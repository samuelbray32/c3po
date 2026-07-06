import os
import numpy as np
import pandas as pd

os.chdir("/home/xulu/code/alison_spyglass")
from alison_behav import *
import datajoint as dj
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.common import AnalysisNwbfile
behavior_model_table = dj.FreeTable(
    dj.conn(), "`alison_rlmodel`.`__behavior_model_results__by_day`"
)
from spyglass.common import Nwbfile
from pynwb import NWBHDF5IO
from spyglass.utils.nwb_helper_fn import get_nwb_file
from spyglass.utils.dj_helper_fn import _get_nwb_object
from spyglass.common import DIOEvents, Nwbfile, interval_list_intersect
import numpy as np


subject = "wilbur"
date = "20210404"
file_name = subject + date
behavior_model_params_name = "default_hmm"
behavior_model_params_name = "default_betalik_addleaf"

def fetch_behavioral_data(
    nwb_file_name,
    behavior_model_params_name= "default_betalik_addleaf",
    trials_info_by_rat_params_name = "decay_default"
):
    file_name = nwb_file_name.split("_.nwb")[0]
    subject = file_name[:-8]
    date = file_name[-8:]

    behav_results = behavior_model_table & {
        "nwb_file_name": nwb_file_name,
        "behavior_model_params_name": behavior_model_params_name,
        # "trials_info_by_rat_params_name": trials_info_by_rat_params_name,
    }
    # all_behav = behav_results.fetch1_dataframe()
    all_behav = fetch_nwb(behav_results, (AnalysisNwbfile, "analysis_file_abs_path"))[0][
        "behavior_model_results_by_day"
    ]
    all_behav


    script_table = StateScriptTrials() & {"nwb_file_name": nwb_file_name}
    trials = []
    epoch = []
    t_poke = []
    t_out = []
    for epoch_i in script_table.fetch("epoch"):
        table = script_table & {"epoch": epoch_i}
        trials.extend(table.fetch1("trial"))
        epoch.extend([epoch_i] * len(table.fetch1("trial")))
        t_poke.extend(table.fetch1("poke_in_ts"))
        t_out.extend(table.fetch1("poke_out_ts"))
    script_df = pd.DataFrame(
        {"epoch": epoch, "trial_number_by_epoch": trials, "t_poke": t_poke, "t_out": t_out},
    )
    script_df

    ## Calculate turn direction of each trial and add to table
    trial_df = pd.merge(all_behav, script_df, on=["epoch", "trial_number_by_epoch"])
    turn_lookup = {
        (1, 2): "right",
        (2, 1): "left",
        (3, 4): "right",
        (4, 3): "left",
        (5, 6): "right",
        (6, 5): "left",
    }

    fill_entries = [
        (1, "left", [3, 4, 5, 6]),
        (2, "right", [3, 4, 5, 6]),
        (3, "left", [1, 2, 5, 6]),
        (4, "right", [1, 2, 5, 6]),
        (5, "left", [1, 2, 3, 4]),
        (6, "right", [1, 2, 3, 4]),
    ]
    for fill_entry in fill_entries:
        port, turn, other_ports = fill_entry
        for other_port in other_ports:
            turn_lookup[(port, other_port)] = turn

    turns = []
    for epoch in trial_df["epoch"].unique():
        epoch_df = trial_df[trial_df["epoch"] == epoch]
        turns.append("start")
        for i in range(1, len(epoch_df)):
            turn = turn_lookup[(epoch_df["leaf"].values[i - 1], epoch_df["leaf"].values[i])]
            turns.append(turn)
    if "turns" not in trial_df.columns:
        trial_df["turns"] = turns

    ## Calculate previous trial reward and add to table
    prev_trial_reward = []
    for epoch in trial_df["epoch"].unique():
        epoch_df = trial_df[trial_df["epoch"] == epoch]
        prev_trial_reward.append(np.nan)
        prev_trial_reward.extend(epoch_df["reward"].values[:-1].tolist())
    if "prev_trial_reward" not in trial_df.columns:
        trial_df["prev_trial_reward"] = prev_trial_reward
    ## Calculate run start times and add to table
    run_starts = []
    for epoch in trial_df["epoch"].unique():
        epoch_df = trial_df[trial_df["epoch"] == epoch]
        run_starts.append(np.nan)
        run_starts.extend(epoch_df["t_out"].values[:-1].tolist())
    if "run_start" not in trial_df.columns:
        trial_df["run_start"] = run_starts


    # load alison data
    alison_df = fetch_alison_df(nwb_file_name)

    # use alisons data to mark segment switches, etc
    actual_segment = alison_df.actual_segment.values
    seg_switch_points = np.where(actual_segment[:-1] != actual_segment[1:])[0] + 1
    from_first_seg = alison_df.is_first_seg_of_trial.iloc[seg_switch_points - 1].values
    first_choice_point_inds = seg_switch_points[from_first_seg]
    first_choice_point_inds
    choice_point_times = alison_df.index.values[first_choice_point_inds]
    len(choice_point_times)

    # find time animal crosses choice point and store
    trial_t_choice = []
    for st, en in zip(trial_df["run_start"], trial_df["t_poke"]):
        ind = np.where((choice_point_times >= st) & (choice_point_times <= en))[0]
        if len(ind) == 1:
            trial_t_choice.append(choice_point_times[ind[0]])
        elif len(ind) == 0:
            trial_t_choice.append(np.nan)
        else:
            raise ValueError("Multiple choice points found in a trial!")
    len(trial_t_choice)
    if "t_choice" not in trial_df.columns:
        trial_df["t_choice"] = trial_t_choice

    # calculate Q values for each trial
    Q_array = trial_df[[f"Q{i}" for i in range(1, 7)]].values
    Q_trial = np.array(
        [Q_array[i, trial_df.leaf.values[i] - 1] for i in range(len(trial_df))]
    )
    Q_trial.shape
    if "Q_trial" not in trial_df.columns:
        trial_df["Q_trial"] = Q_trial
    Q_stem_array = trial_df[[f"Qstem{i}" for i in range(1, 4)]].values
    stem_dict = {"A": 0, "B": 1, "C": 2}
    trial_stem_index = np.array([stem_dict[s] for s in trial_df.stem.values])
    Q_stem_trial = np.array(
        [Q_stem_array[i, trial_stem_index[i]] for i in range(len(trial_df))]
    )
    if not "Q_stem_trial" in trial_df.columns:
        trial_df["Q_stem_trial"] = Q_stem_trial
    Q_stem_max = Q_stem_array.max(axis=1)
    switch_val = Q_stem_max - Q_stem_trial
    if not "switch_val" in trial_df.columns:
        trial_df["switch_val"] = switch_val

    # Tina's Q_A / Q_B nomenclature
    # =========================
    n = len(trial_df)
    leaf = trial_df["leaf"].values
    reward = trial_df["reward"].values

    last_visit_idx = np.full(n, np.nan)
    last_seen = {}

    for i in range(n):
        l = leaf[i]
        if l in last_seen:
            last_visit_idx[i] = last_seen[l]
        last_seen[l] = i

    trial_df["last_visit_idx"] = last_visit_idx

    # =========================
    # STEP 3: LEAF B (current leaf) & LEAF C (next leaf, current value)
    # =========================

    Q_leaf_B = []
    Q_leaf_B_prev = []
    Q_leaf_B_next = []
    Q_leaf_C = []  # <--- NEW: List to hold values for the next chosen leaf
    Leaf_B_prev_reward = []
    Q_leaf_B_prev_update = []

    for i in range(n):
        l = leaf[i]

        # --- Q_leaf_B (trial 0) ---
        Q_array_current = Q_array[i]  # Current trial's Q-values
        Q_leaf_B.append(Q_array_current[l - 1])

        # --- Q_leaf_C (Next trial's leaf, current trial's Q-value) ---
        if i + 1 < n:
            next_leaf = int(leaf[i + 1])
            Q_leaf_C.append(Q_array_current[next_leaf - 1])  # Peek at next leaf, pull from current Q-array
        else:
            Q_leaf_C.append(np.nan)  # Last trial has no "next trial"

        # --- last visit (trial -n) ---
        prev_i = last_visit_idx[i]

        if np.isnan(prev_i):
            Q_leaf_B_prev.append(np.nan)
            Leaf_B_prev_reward.append(np.nan)
            Q_leaf_B_prev_update.append(np.nan)
        else:
            prev_i = int(prev_i)

            # Q at last visit (trial -n)
            Q_leaf_B_prev.append(Q_array[prev_i, l - 1])

            # reward at last visit
            Leaf_B_prev_reward.append(reward[prev_i])

            # update from that visit:
            # Q(trial -(n-1)) - Q(trial -n)
            if prev_i + 1 < n:
                Q_after = Q_array[prev_i + 1, l - 1]
                Q_before = Q_array[prev_i, l - 1]
                Q_leaf_B_prev_update.append(Q_after - Q_before)
            else:
                Q_leaf_B_prev_update.append(np.nan)

        # --- Q_leaf_B_next (trial +1) ---
        if i + 1 < n:
            Q_leaf_B_next.append(Q_array[i + 1, l - 1])
        else:
            Q_leaf_B_next.append(np.nan)

    # Assign arrays back to your dataframe
    trial_df["Q_leaf_B"] = Q_leaf_B
    trial_df["Q_leaf_B_prev"] = Q_leaf_B_prev
    trial_df["Q_leaf_B_next"] = Q_leaf_B_next
    trial_df["Q_leaf_C"] = Q_leaf_C  # <--- NEW: Save to dataframe

    trial_df["Leaf_B_reward"] = reward
    trial_df["Leaf_B_prev_reward"] = Leaf_B_prev_reward

    trial_df["Q_leaf_B_prev_update"] = Q_leaf_B_prev_update
    trial_df["Q_leaf_B_update"] = trial_df["Q_leaf_B_next"] - trial_df["Q_leaf_B"]

    # =========================
    # STEP 4: LEAF A (previous leaf)
    # =========================

    prev_leaf = trial_df["leaf"].shift(1).values

    Q_leaf_A_prev = []
    Q_leaf_A = []

    for i in range(n):

        if i == 0 or np.isnan(prev_leaf[i]):
            Q_leaf_A_prev.append(np.nan)
            Q_leaf_A.append(np.nan)
            continue

        l_prev = int(prev_leaf[i])

        # Q at trial -1 (before update)
        Q_leaf_A_prev.append(Q_array[i - 1, l_prev - 1])

        # Q at trial 0 (after update)
        Q_leaf_A.append(Q_array[i, l_prev - 1])

    trial_df["Q_leaf_A_prev"] = Q_leaf_A_prev
    trial_df["Q_leaf_A"] = Q_leaf_A

    trial_df["Leaf_A_reward"] = trial_df["reward"].shift(1)
    trial_df["Q_leaf_A_update"] = trial_df["Q_leaf_A"] - trial_df["Q_leaf_A_prev"]

    # =========================
    # STEP 6: Previous Poke Time
    # =========================
    t_poke_prev = trial_df.t_poke.values[:-1]
    t_poke_prev = np.insert(t_poke_prev, 0, np.nan)
    trial_df["t_poke_prev"] = t_poke_prev

    # =========================
    # STEP 7: is this a pre_switch trial?
    # =========================
    trial_df["before_switch"] = (trial_df["n_pre_switch"] == 1).astype(int)

    # =========================
    # STEP 8: Overall Trial Number
    # =========================
    trial_df["trial_num"] = trial_df.index.values

    # =========================
    # STEP 9: Q dep values
    # =========================
    Qdep_array = trial_df[[f"Qdep{i}" for i in range(1, 7)]].values

    leaf = trial_df["leaf"].values
    reward = trial_df["reward"].values
    n = len(trial_df)


    # =========================
    # STEP 9.1: LEAF B (current leaf)
    # =========================

    Qdep_leaf_B = []
    Qdep_leaf_B_prev = []
    Qdep_leaf_B_next = []
    Qdep_leaf_B_prev_update = []

    for i in range(n):
        l = leaf[i]

        # --- Q_leaf_B (trial 0) ---
        Qdep_leaf_B.append(Qdep_array[i, l - 1])

        # --- last visit (trial -n) ---
        prev_i = last_visit_idx[i]

        if np.isnan(prev_i):
            Qdep_leaf_B_prev.append(np.nan)
            Qdep_leaf_B_prev_update.append(np.nan)
        else:
            prev_i = int(prev_i)

            # Q at last visit (trial -n)
            Qdep_leaf_B_prev.append(Qdep_array[prev_i, l - 1])

            # update from that visit:
            # Q(trial -(n-1)) - Q(trial -n)
            if prev_i + 1 < n:
                Qdep_after = Qdep_array[prev_i + 1, l - 1]
                Qdep_before = Qdep_array[prev_i, l - 1]
                Qdep_leaf_B_prev_update.append(Qdep_after - Qdep_before)
            else:
                Qdep_leaf_B_prev_update.append(np.nan)

        # --- Q_leaf_B_next (trial +1) ---
        if i + 1 < n:
            Qdep_leaf_B_next.append(Qdep_array[i + 1, l - 1])
        else:
            Qdep_leaf_B_next.append(np.nan)

    trial_df["Qdep_leaf_B"] = Qdep_leaf_B
    trial_df["Qdep_leaf_B_prev"] = Qdep_leaf_B_prev
    trial_df["Qdep_leaf_B_next"] = Qdep_leaf_B_next

    trial_df["Qdep_leaf_B_prev_update"] = Qdep_leaf_B_prev_update
    trial_df["Qdep_leaf_B_new_update"] = trial_df["Qdep_leaf_B_next"] - trial_df["Qdep_leaf_B"]

    # =========================
    # STEP 9.2: LEAF A (previous leaf)
    # =========================

    prev_leaf = trial_df["leaf"].shift(1).values

    Qdep_leaf_A_prev = []
    Qdep_leaf_A = []

    for i in range(n):

        if i == 0 or np.isnan(prev_leaf[i]):
            Qdep_leaf_A_prev.append(np.nan)
            Qdep_leaf_A.append(np.nan)
            continue

        l_prev = int(prev_leaf[i])

        # Q at trial -1 (before update)
        Qdep_leaf_A_prev.append(Qdep_array[i - 1, l_prev - 1])

        # Q at trial 0 (after update)
        Qdep_leaf_A.append(Qdep_array[i, l_prev - 1])

    trial_df["Qdep_leaf_A_prev"] = Qdep_leaf_A_prev
    trial_df["Qdep_leaf_A"] = Qdep_leaf_A

    trial_df["Qdep_leaf_A_update"] = trial_df["Qdep_leaf_A"] - trial_df["Qdep_leaf_A_prev"]


    return {"trial_df": trial_df,
            "alison_df": alison_df}



def fetch_dio_times(nwb_file_name):
    # Fetch the NWB file and DIO pump times
    key = {"nwb_file_name": nwb_file_name}
    nwb = (Nwbfile() & key).fetch_nwb()[0]
    dio_query = (
        DIOEvents()
        & key
        & "dio_event_name LIKE 'pump%'"
    )

    pump_obj_ids = dio_query.fetch("dio_object_id")
    all_pump_intervals = []
    for obj_id in pump_obj_ids:
        obj = _get_nwb_object(nwb.objects, obj_id)
        state = obj.data[:]
        state_time = obj.timestamps[:]
        ind_on = np.where(state == 1)[0]
        pump_intervals = [
            [st, en] for st, en in zip(state_time[ind_on], state_time[ind_on + 1])
        ]
        all_pump_intervals.extend(pump_intervals)
    all_pump_intervals = np.array(all_pump_intervals)
    ind = np.argsort(all_pump_intervals[:, 0])
    all_pump_intervals = all_pump_intervals[ind] # return this

    dio_query = (
        DIOEvents()
        & key
        & "dio_event_name LIKE 'poke%'"
    )
    poke_obj_ids = dio_query.fetch("dio_object_id")
    all_poke_intervals = []
    for obj_id in poke_obj_ids:
        obj = _get_nwb_object(nwb.objects, obj_id)
        state = obj.data[:]
        state_time = obj.timestamps[:]
        ind_on = np.where(state == 1)[0]
        poke_intervals = [
            [st, en] for st, en in zip(state_time[ind_on], state_time[ind_on + 1])
        ]
        all_poke_intervals.extend(poke_intervals)
    all_poke_intervals = np.array(all_poke_intervals)
    ind = np.argsort(all_poke_intervals[:, 0])
    all_poke_intervals = all_poke_intervals[ind] # return this
    return {
        "all_pump_intervals": all_pump_intervals,
        "all_poke_intervals": all_poke_intervals
    }

def fetch_decoding_df(nwb_file_name):
    key = {"nwb_file_name": nwb_file_name}
    schema = dj.schema("alison_decoding")
    table_name = "`alison_decoding`.`__clusterless_acausal_results_summary`"
    acausal_results_table = dj.FreeTable(dj.conn(), table_name)

    acausal_results_keys = (acausal_results_table & key).fetch("KEY")
    acausal_results = acausal_results_table & acausal_results_keys
    nwb_list = fetch_nwb(acausal_results, (AnalysisNwbfile, "analysis_file_abs_path"))
    decoding_df = pd.concat([nwb["results_df"] for nwb in nwb_list], ignore_index=True)
    decoding_df.set_index("time", inplace=True)
    return decoding_df

def fetch_alison_df(nwb_file_name):
    file_name = nwb_file_name.split("_.nwb")[0]
    subject_id = file_name[:-8]
    date = file_name[-8:]

    # quick loading extracted df
    out_path = f"/stelmo/sam/c3po_datasets/{subject_id}{date}_big_df.pkl"
    if os.path.exists(out_path):
        return pd.read_pickle(out_path)

    #slow loading from the big df
    out_path = "/stelmo/alison/big_df_pkls/"
    today_now = "20230113"
    wilbur_big_df = pd.read_pickle(
        out_path + subject_id + "_big_df_stabledecayNoRL_" + today_now + ".pkl"
    )
    return wilbur_big_df[wilbur_big_df["nwb_file_name"] == nwb_file_name]


