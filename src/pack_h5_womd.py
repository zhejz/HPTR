# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import sys

sys.path.append(".")

from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import numpy as np
from pathlib import Path
from waymo_open_dataset.protos import scenario_pb2
import src.utils.pack_h5 as pack_utils
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# "map/type"
# FREEWAY = 0
# SURFACE_STREET = 1
# STOP_SIGN = 2
# BIKE_LANE = 3
# TYPE_ROAD_EDGE_BOUNDARY = 4
# TYPE_ROAD_EDGE_MEDIAN = 5
# BROKEN = 6
# SOLID_SINGLE = 7
# DOUBLE = 8
# SPEED_BUMP = 9
# CROSSWALK = 10
N_PL_TYPE = 11
DIM_VEH_LANES = [0, 1, 2]
DIM_CYC_LANES = [3]
DIM_PED_LANES = [4]

# "tl_lane/state", "tl_stop/state"
# LANE_STATE_UNKNOWN = 0;
# LANE_STATE_STOP = 1;
# LANE_STATE_CAUTION = 2;
# LANE_STATE_GO = 3;
# LANE_STATE_FLASHING = 4;
N_TL_STATE = 5

N_PL_MAX = 3000
N_TL_MAX = 40
N_AGENT_MAX = 1300

N_PL = 1024
N_TL = 100
N_AGENT = 64
N_AGENT_NO_SIM = 256

THRESH_MAP = 120
THRESH_AGENT = 120

N_STEP = 91
STEP_CURRENT = 10


def collate_agent_features(tracks, sdc_track_index, track_index_predict, object_id_interest):
    agent_id = []
    agent_type = []
    agent_states = []
    agent_role = []
    for i, _track in enumerate(tracks):
        agent_id.append(_track.id)
        agent_type.append(_track.object_type - 1)  # [TYPE_VEHICLE=1, TYPE_PEDESTRIAN=2, TYPE_CYCLIST=3] -> [0,1,2]
        step_states = []
        for s in _track.states:
            step_states.append(
                [
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.length,
                    s.width,
                    s.height,
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                    s.valid,
                ]
            )
            # This angle is normalized to [-pi, pi). The velocity vector in m/s
        agent_states.append(step_states)

        agent_role.append([False, False, False])
        if i in track_index_predict:
            agent_role[-1][2] = True
        if _track.id in object_id_interest:
            agent_role[-1][1] = True
        if i == sdc_track_index:
            agent_role[-1][0] = True

    return agent_id, agent_type, agent_states, agent_role


def collate_tl_features(tl_features):
    tl_lane_state = []
    tl_lane_id = []
    tl_stop_point = []
    for _step_tl in tl_features:
        step_tl_lane_state = []
        step_tl_lane_id = []
        step_tl_stop_point = []
        for _tl in _step_tl.lane_states:
            if _tl.state == 0:  # LANE_STATE_UNKNOWN = 0;
                tl_state = 0  # LANE_STATE_UNKNOWN = 0;
            elif _tl.state in [1, 4]:  # LANE_STATE_ARROW_STOP = 1; LANE_STATE_STOP = 4;
                tl_state = 1  # LANE_STATE_STOP = 1;
            elif _tl.state in [2, 5]:  # LANE_STATE_ARROW_CAUTION = 2; LANE_STATE_CAUTION = 5;
                tl_state = 2  # LANE_STATE_CAUTION = 2;
            elif _tl.state in [3, 6]:  # LANE_STATE_ARROW_GO = 3; LANE_STATE_GO = 6;
                tl_state = 3  # LANE_STATE_GO = 3;
            elif _tl.state in [7, 8]:  # LANE_STATE_FLASHING_STOP = 7; LANE_STATE_FLASHING_CAUTION = 8;
                tl_state = 4  # LANE_STATE_FLASHING = 4;
            else:
                assert ValueError

            step_tl_lane_state.append(tl_state)
            step_tl_lane_id.append(_tl.lane)
            step_tl_stop_point.append([_tl.stop_point.x, _tl.stop_point.y, _tl.stop_point.z])

        tl_lane_state.append(step_tl_lane_state)
        tl_lane_id.append(step_tl_lane_id)
        tl_stop_point.append(step_tl_stop_point)
    return tl_lane_state, tl_lane_id, tl_stop_point


def collate_map_features(map_features):
    mf_id = []
    mf_xyz = []
    mf_type = []
    mf_edge = []
    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        if feature_data_type is None:  # pip install waymo-open-dataset-tf-2-6-0==1.4.9, not updated, should be driveway
            continue
        feature = getattr(mf, feature_data_type)
        if feature_data_type == "lane":
            if feature.type == 0:  # UNDEFINED
                mf_type.append(1)
            elif feature.type == 1:  # FREEWAY
                mf_type.append(0)
            elif feature.type == 2:  # SURFACE_STREET
                mf_type.append(1)
            elif feature.type == 3:  # BIKE_LANE
                mf_type.append(3)
            mf_id.append(mf.id)
            mf_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])
            if len(feature.exit_lanes) > 0:
                for _id_exit in feature.exit_lanes:
                    mf_edge.append([mf.id, _id_exit])
            else:
                mf_edge.append([mf.id, -1])
        elif feature_data_type == "stop_sign":
            for l_id in feature.lane:
                # override FREEWAY/SURFACE_STREET with stop sign lane
                # BIKE_LANE remains unchanged
                idx_lane = mf_id.index(l_id)
                if mf_type[idx_lane] < 2:
                    mf_type[idx_lane] = 2
        elif feature_data_type == "road_edge":
            assert feature.type > 0  # no UNKNOWN = 0
            mf_id.append(mf.id)
            mf_type.append(feature.type + 3)  # [1, 2] -> [4, 5]
            mf_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])
        elif feature_data_type == "road_line":
            assert feature.type > 0  # no UNKNOWN = 0
            # BROKEN_SINGLE_WHITE = 1
            # SOLID_SINGLE_WHITE = 2
            # SOLID_DOUBLE_WHITE = 3
            # BROKEN_SINGLE_YELLOW = 4
            # BROKEN_DOUBLE_YELLOW = 5
            # SOLID_SINGLE_YELLOW = 6
            # SOLID_DOUBLE_YELLOW = 7
            # PASSING_DOUBLE_YELLOW = 8
            if feature.type in [1, 4, 5]:
                feature_type_new = 6  # BROKEN
            elif feature.type in [2, 6]:
                feature_type_new = 7  # SOLID_SINGLE
            else:
                feature_type_new = 8  # DOUBLE
            mf_id.append(mf.id)
            mf_type.append(feature_type_new)
            mf_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])
        elif feature_data_type == "speed_bump" or feature_data_type == "crosswalk":
            xyz = np.array([[p.x, p.y, p.z] for p in feature.polygon])
            polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)
            pl_polygon = pack_utils.get_polylines_from_polygon(xyz[polygon_idx])
            mf_xyz.extend(pl_polygon)
            mf_id.extend([mf.id] * len(pl_polygon))
            pl_type = 9 if feature_data_type == "speed_bump" else 10
            mf_type.extend([pl_type] * len(pl_polygon))

    return mf_id, mf_xyz, mf_type, mf_edge


def main():
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir", default="/cluster/scratch/zhejzhan/womd_scenario_v_1_2_0")
    parser.add_argument("--dataset", default="training")
    parser.add_argument("--out-dir", default="/cluster/scratch/zhejzhan/h5_womd_hptr")
    parser.add_argument("--rand-pos", default=50.0, type=float, help="Meter. Set to -1 to disable.")
    parser.add_argument("--rand-yaw", default=3.14, type=float, help="Radian. Set to -1 to disable.")
    parser.add_argument("--dest-no-pred", action="store_true")
    args = parser.parse_args()

    dataset_size = {
        "training": 486995,  # 487002
        "validation": 44097,
        "training_20s": 70541,
        "validation_interactive": 43479,
        "testing": 44920,
        "testing_interactive": 44154,
    }

    if "training" in args.dataset:
        pack_all = True  # ["agent/valid"]
        pack_history = False  # ["history/agent/valid"]
    elif "validation" in args.dataset:
        pack_all = True
        pack_history = True
    elif "testing" in args.dataset:
        pack_all = False
        pack_history = True

    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    out_h5_path = out_path / (args.dataset + ".h5")

    data_path = Path(args.data_dir) / args.dataset
    dataset = tf.data.TFRecordDataset(sorted([p.as_posix() for p in data_path.glob("*")]), compression_type="")
    n_pl_max, n_tl_max, n_agent_max, n_agent_sim, n_agent_no_sim, data_len = 0, 0, 0, 0, 0, 0
    with h5py.File(out_h5_path, "w") as hf:
        for i, data in tqdm(enumerate(dataset), total=dataset_size[args.dataset]):

            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())

            mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(scenario.map_features)
            tl_lane_state, tl_lane_id, tl_stop_point = collate_tl_features(scenario.dynamic_map_states)
            agent_id, agent_type, agent_states, agent_role = collate_agent_features(
                scenario.tracks,
                sdc_track_index=scenario.sdc_track_index,
                track_index_predict=[i.track_index for i in scenario.tracks_to_predict],
                object_id_interest=[i for i in scenario.objects_of_interest],
            )

            episode = {}
            n_pl = pack_utils.pack_episode_map(
                episode=episode, mf_id=mf_id, mf_xyz=mf_xyz, mf_type=mf_type, mf_edge=mf_edge, n_pl_max=N_PL_MAX
            )
            n_tl = pack_utils.pack_episode_traffic_lights(
                episode=episode,
                tl_lane_state=tl_lane_state,
                tl_lane_id=tl_lane_id,
                tl_stop_point=tl_stop_point,
                pack_all=pack_all,
                pack_history=pack_history,
                n_tl_max=N_TL_MAX,
                step_current=STEP_CURRENT,
            )
            n_agent = pack_utils.pack_episode_agents(
                episode=episode,
                agent_id=agent_id,
                agent_type=agent_type,
                agent_states=agent_states,
                agent_role=agent_role,
                pack_all=pack_all,
                pack_history=pack_history,
                n_agent_max=N_AGENT_MAX,
                step_current=STEP_CURRENT,
            )
            scenario_center, scenario_yaw = pack_utils.center_at_sdc(episode, args.rand_pos, args.rand_yaw)
            n_pl_max = max(n_pl_max, n_pl)
            n_tl_max = max(n_tl_max, n_tl)
            n_agent_max = max(n_agent_max, n_agent)

            episode_reduced = {}
            pack_utils.filter_episode_map(episode, N_PL, THRESH_MAP, thresh_z=3)
            episode_with_map = episode["map/valid"].any(1).sum() > 0
            pack_utils.repack_episode_map(episode, episode_reduced, N_PL, N_PL_TYPE)

            pack_utils.filter_episode_traffic_lights(episode)
            pack_utils.repack_episode_traffic_lights(episode, episode_reduced, N_TL, N_TL_STATE)

            if "training" in args.dataset:
                mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
                    episode=episode,
                    episode_reduced=episode_reduced,
                    n_agent=N_AGENT,
                    prefix="",
                    dim_veh_lanes=DIM_VEH_LANES,
                    dist_thresh_agent=THRESH_AGENT,
                    step_current=STEP_CURRENT,
                )
                pack_utils.repack_episode_agents(
                    episode=episode,
                    episode_reduced=episode_reduced,
                    mask_sim=mask_sim,
                    n_agent=N_AGENT,
                    prefix="",
                    dim_veh_lanes=DIM_VEH_LANES,
                    dim_cyc_lanes=DIM_CYC_LANES,
                    dim_ped_lanes=DIM_PED_LANES,
                    dest_no_pred=args.dest_no_pred,
                )
            elif "validation" in args.dataset:
                mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
                    episode=episode,
                    episode_reduced=episode_reduced,
                    n_agent=N_AGENT,
                    prefix="history/",
                    dim_veh_lanes=DIM_VEH_LANES,
                    dist_thresh_agent=THRESH_AGENT,
                    step_current=STEP_CURRENT,
                )
                pack_utils.repack_episode_agents(
                    episode=episode,
                    episode_reduced=episode_reduced,
                    mask_sim=mask_sim,
                    n_agent=N_AGENT,
                    prefix="",
                    dim_veh_lanes=DIM_VEH_LANES,
                    dim_cyc_lanes=DIM_CYC_LANES,
                    dim_ped_lanes=DIM_PED_LANES,
                    dest_no_pred=args.dest_no_pred,
                )
                pack_utils.repack_episode_agents(episode, episode_reduced, mask_sim, N_AGENT, "history/")
                pack_utils.repack_episode_agents_no_sim(episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "")
                pack_utils.repack_episode_agents_no_sim(
                    episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
                )
            elif "testing" in args.dataset:
                if episode_with_map:
                    mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
                        episode=episode,
                        episode_reduced=episode_reduced,
                        n_agent=N_AGENT,
                        prefix="history/",
                        dim_veh_lanes=DIM_VEH_LANES,
                        dist_thresh_agent=THRESH_AGENT,
                        step_current=STEP_CURRENT,
                    )
                else:
                    mask_valid = episode["history/agent/valid"].any(0)
                    mask_sim = episode["history/agent/role"].any(-1)
                    for _valid_idx in np.where(mask_valid)[0]:
                        mask_sim[_valid_idx] = True
                        if mask_sim.sum() >= N_AGENT:
                            break
                    mask_no_sim = mask_valid & (~mask_sim)

                pack_utils.repack_episode_agents(episode, episode_reduced, mask_sim, N_AGENT, "history/")
                pack_utils.repack_episode_agents_no_sim(
                    episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
                )
            n_agent_sim = max(n_agent_sim, mask_sim.sum())
            n_agent_no_sim = max(n_agent_no_sim, mask_no_sim.sum())

            if episode_with_map:
                episode_reduced["map/boundary"] = pack_utils.get_map_boundary(
                    episode_reduced["map/valid"], episode_reduced["map/pos"]
                )
            else:
                # only in waymo test split.
                assert args.dataset == "testing"
                episode_reduced["map/boundary"] = pack_utils.get_map_boundary(
                    episode["history/agent/valid"], episode["history/agent/pos"]
                )
                print(f"scenario {i} has no map! map boundary is: {episode_reduced['map/boundary']}")

            hf_episode = hf.create_group(str(i))
            hf_episode.attrs["scenario_id"] = scenario.scenario_id
            hf_episode.attrs["scenario_center"] = scenario_center
            hf_episode.attrs["scenario_yaw"] = scenario_yaw
            hf_episode.attrs["with_map"] = episode_with_map

            for k, v in episode_reduced.items():
                hf_episode.create_dataset(k, data=v, compression="gzip", compression_opts=4, shuffle=True)

            data_len += 1

        print(f"data_len: {data_len}, dataset_size: {dataset_size[args.dataset]}")
        print(f"n_pl_max: {n_pl_max}")
        print(f"n_agent_max: {n_agent_max}, n_agent_sim: {n_agent_sim}, n_agent_no_sim: {n_agent_no_sim}")
        hf.attrs["data_len"] = data_len


if __name__ == "__main__":
    main()
