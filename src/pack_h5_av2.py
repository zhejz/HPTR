# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import sys

sys.path.append(".")

from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.interpolate import interp_arc
import src.utils.pack_h5 as pack_utils

# "map/type"
# VEHICLE = 0
# BUS = 1
# BIKE = 2
# UNKNOWN = 3
# DOUBLE_DASH = 4
# DASHED = 5
# SOLID = 6
# DOUBLE_SOLID = 7
# DASH_SOLID = 8
# SOLID_DASH = 9
# CROSSWALK = 10
N_PL_TYPE = 11
PL_TYPE = {
    "VEHICLE": 0,
    "BUS": 1,
    "BIKE": 2,
    "NONE": 3,
    "UNKNOWN": 3,
    "DOUBLE_DASH_WHITE": 4,
    "DOUBLE_DASH_YELLOW": 4,
    "DASHED_WHITE": 5,
    "DASHED_YELLOW": 5,
    "SOLID_WHITE": 6,
    "SOLID_BLUE": 6,
    "SOLID_YELLOW": 6,
    "DOUBLE_SOLID_WHITE": 7,
    "DOUBLE_SOLID_YELLOW": 7,
    "DASH_SOLID_YELLOW": 8,
    "DASH_SOLID_WHITE": 8,
    "SOLID_DASH_WHITE": 9,
    "SOLID_DASH_YELLOW": 9,
    "CROSSWALK": 10,
}
DIM_VEH_LANES = [0, 1]
DIM_CYC_LANES = [2]
DIM_PED_LANES = [3, 6, 10]

# "agent/type"
AGENT_TYPE = {
    "vehicle": 0,
    "bus": 0,
    "pedestrian": 1,
    "motorcyclist": 2,
    "cyclist": 2,
    "riderless_bicycle": 2,
}
AGENT_SIZE = {
    "vehicle": [4.7, 2.1, 1.7],
    "bus": [11, 3, 3.5],
    "pedestrian": [0.85, 0.85, 1.75],
    "motorcyclist": [2, 0.8, 1.8],
    "cyclist": [2, 0.8, 1.8],
    "riderless_bicycle": [2, 0.8, 1.8],
}


N_PL_MAX = 1500
N_AGENT_MAX = 256

N_PL = 1024
N_AGENT = 64
N_AGENT_NO_SIM = N_AGENT_MAX - N_AGENT

THRESH_MAP = 120
THRESH_AGENT = 120

STEP_CURRENT = 49
N_STEP = 110


def collate_agent_features(scenario_path, n_step):
    tracks = scenario_serialization.load_argoverse_scenario_parquet(scenario_path).tracks

    agent_id = []
    agent_type = []
    agent_states = []
    agent_role = []
    for _track in tracks:
        if _track.object_type not in AGENT_TYPE:
            continue
        # role: [sdc=0, interest=1, predict=2]
        if _track.track_id == "AV":
            agent_id.append(0)
            agent_role.append([True, False, False])
        else:
            assert int(_track.track_id) > 0
            agent_id.append(int(_track.track_id))
            # [sdc=0, interest=1, predict=2]
            if _track.category.value == 2:  # SCORED_TRACK
                agent_role.append([False, True, False])
            elif _track.category.value == 3:  # FOCAL_TRACK
                agent_role.append([False, False, True])
            else:  # TRACK_FRAGMENT or UNSCORED_TRACK
                agent_role.append([False, False, False])

        agent_type.append(AGENT_TYPE[_track.object_type])

        step_states = [[0.0] * 9 + [False]] * n_step
        agent_size = AGENT_SIZE[_track.object_type]
        for s in _track.object_states:
            step_states[int(s.timestep)] = [
                s.position[0],  # center_x
                s.position[1],  # center_y
                3,  # center_z
                agent_size[0],  # length
                agent_size[1],  # width
                agent_size[2],  # height
                s.heading,  # heading in radian
                s.velocity[0],  # velocity_x
                s.velocity[1],  # velocity_y
                True,  # valid
            ]

        agent_states.append(step_states)

    return agent_id, agent_type, agent_states, agent_role


def collate_map_features(map_path):
    static_map = ArgoverseStaticMap.from_json(map_path)

    def _interpolate_centerline(left_ln_boundary, right_ln_boundary):
        num_interp_pts = (
            np.linalg.norm(np.diff(left_ln_boundary, axis=0), axis=-1).sum()
            + np.linalg.norm(np.diff(right_ln_boundary, axis=0), axis=-1).sum()
        ) / 2.0
        num_interp_pts = int(num_interp_pts) + 1
        left_even_pts = interp_arc(num_interp_pts, points=left_ln_boundary)
        right_even_pts = interp_arc(num_interp_pts, points=right_ln_boundary)
        centerline_pts = (left_even_pts + right_even_pts) / 2.0
        return centerline_pts, left_even_pts, right_even_pts

    mf_id = []
    mf_xyz = []
    mf_type = []
    mf_edge = []
    lane_boundary_set = []

    for _id, ped_xing in static_map.vector_pedestrian_crossings.items():
        v0, v1 = ped_xing.edge1.xyz
        v2, v3 = ped_xing.edge2.xyz
        pl_crosswalk = pack_utils.get_polylines_from_polygon(np.array([v0, v1, v3, v2]))
        mf_id.extend([_id] * len(pl_crosswalk))
        mf_type.extend([PL_TYPE["CROSSWALK"]] * len(pl_crosswalk))
        mf_xyz.extend(pl_crosswalk)

    for _id, lane_segment in static_map.vector_lane_segments.items():
        centerline_pts, left_even_pts, right_even_pts = _interpolate_centerline(
            lane_segment.left_lane_boundary.xyz, lane_segment.right_lane_boundary.xyz
        )

        mf_id.append(_id)
        mf_xyz.append(centerline_pts)
        mf_type.append(PL_TYPE[lane_segment.lane_type])

        if (lane_segment.left_lane_boundary not in lane_boundary_set) and not (
            lane_segment.is_intersection and lane_segment.left_mark_type in ["NONE", "UNKOWN"]
        ):
            lane_boundary_set.append(lane_segment.left_lane_boundary)
            mf_xyz.append(left_even_pts)
            mf_id.append(-2)
            mf_type.append(PL_TYPE[lane_segment.left_mark_type])

        if (lane_segment.right_lane_boundary not in lane_boundary_set) and not (
            lane_segment.is_intersection and lane_segment.right_mark_type in ["NONE", "UNKOWN"]
        ):
            lane_boundary_set.append(lane_segment.right_lane_boundary)
            mf_xyz.append(right_even_pts)
            mf_id.append(-2)
            mf_type.append(PL_TYPE[lane_segment.right_mark_type])

            for _id_exit in lane_segment.successors:
                mf_edge.append([_id, _id_exit])
        else:
            mf_edge.append([_id, -1])

    return mf_id, mf_xyz, mf_type, mf_edge


def main():
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir", default="/cluster/scratch/zhejzhan/av2_motion")
    parser.add_argument("--dataset", default="training")
    parser.add_argument("--out-dir", default="/cluster/scratch/zhejzhan/h5_av2_hptr")
    parser.add_argument("--rand-pos", default=50.0, type=float, help="Meter. Set to -1 to disable.")
    parser.add_argument("--rand-yaw", default=3.14, type=float, help="Radian. Set to -1 to disable.")
    parser.add_argument("--dest-no-pred", action="store_true")
    args = parser.parse_args()

    if "training" in args.dataset:
        pack_all = True  # ["agent/valid"]
        pack_history = False  # ["history/agent/valid"]
        n_step = N_STEP
    elif "validation" in args.dataset:
        pack_all = True
        pack_history = True
        n_step = N_STEP
    elif "testing" in args.dataset:
        pack_all = False
        pack_history = True
        n_step = STEP_CURRENT + 1

    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    out_h5_path = out_path / (args.dataset + ".h5")

    scenario_list = sorted(list((Path(args.data_dir) / args.dataset).glob("*")))
    n_pl_max, n_agent_max, n_agent_sim, n_agent_no_sim = 0, 0, 0, 0
    with h5py.File(out_h5_path, "w") as hf:
        hf.attrs["data_len"] = len(scenario_list)
        for i in tqdm(range(hf.attrs["data_len"])):
            scenario_folder = scenario_list[i]
            mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(
                scenario_folder / f"log_map_archive_{scenario_folder.name}.json"
            )
            agent_id, agent_type, agent_states, agent_role = collate_agent_features(
                scenario_folder / f"scenario_{scenario_folder.name}.parquet", n_step
            )

            episode = {}
            n_pl = pack_utils.pack_episode_map(
                episode=episode, mf_id=mf_id, mf_xyz=mf_xyz, mf_type=mf_type, mf_edge=mf_edge, n_pl_max=N_PL_MAX
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
            n_agent_max = max(n_agent_max, n_agent)

            episode_reduced = {}
            pack_utils.filter_episode_map(episode, N_PL, THRESH_MAP, thresh_z=-1)
            assert episode["map/valid"].any(1).sum() > 0
            pack_utils.repack_episode_map(episode, episode_reduced, N_PL, N_PL_TYPE)

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
                mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
                    episode=episode,
                    episode_reduced=episode_reduced,
                    n_agent=N_AGENT,
                    prefix="history/",
                    dim_veh_lanes=DIM_VEH_LANES,
                    dist_thresh_agent=THRESH_AGENT,
                    step_current=STEP_CURRENT,
                )
                pack_utils.repack_episode_agents(episode, episode_reduced, mask_sim, N_AGENT, "history/")
                pack_utils.repack_episode_agents_no_sim(
                    episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
                )
            n_agent_sim = max(n_agent_sim, mask_sim.sum())
            n_agent_no_sim = max(n_agent_no_sim, mask_no_sim.sum())

            episode_reduced["map/boundary"] = pack_utils.get_map_boundary(
                episode_reduced["map/valid"], episode_reduced["map/pos"]
            )

            hf_episode = hf.create_group(str(i))
            hf_episode.attrs["scenario_id"] = scenario_folder.name
            hf_episode.attrs["scenario_center"] = scenario_center
            hf_episode.attrs["scenario_yaw"] = scenario_yaw
            hf_episode.attrs["with_map"] = True

            for k, v in episode_reduced.items():
                hf_episode.create_dataset(k, data=v, compression="gzip", compression_opts=4, shuffle=True)

        print(f"n_pl_max: {n_pl_max}, n_agent_max: {n_agent_max}")
        print(f"n_agent_sim: {n_agent_sim}, n_agent_no_sim: {n_agent_no_sim}")


if __name__ == "__main__":
    main()
