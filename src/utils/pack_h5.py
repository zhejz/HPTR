# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Tuple
import numpy as np
from scipy.interpolate import interp1d
from . import transform_utils

# "agent/cmd"
# STATIONARY = 0;
# STRAIGHT = 1;
# STRAIGHT_LEFT = 2;
# STRAIGHT_RIGHT = 3;
# LEFT_U_TURN = 4;
# LEFT_TURN = 5;
# RIGHT_U_TURN = 6;
# RIGHT_TURN = 7;
N_AGENT_CMD = 8


def get_polylines_from_polygon(polygon: np.ndarray) -> List[List[List]]:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> List[List]:
        length = np.linalg.norm(start - end)
        unit_vec = (end - start) / length
        pl = []
        for i in range(int(length) + 1):  # 4.5 -> 5 [0,1,2,3,4]
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])
        pl.append([end[0], end[1], end[2]])
        return pl

    # if l1 > l2:
    #     pl1 = _pl_interp_start_end((polygon[0] + polygon[3]) * 0.5, (polygon[1] + polygon[2]) * 0.5)
    # else:
    #     pl1 = _pl_interp_start_end((polygon[0] + polygon[1]) * 0.5, (polygon[2] + polygon[3]) * 0.5)
    # return [pl1, pl1[::-1]]

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])
    return [pl1, pl1[::-1], pl2, pl2[::-1]]


def get_map_boundary(map_valid, map_pos):
    """
    Args:
        map_valid: [n_pl, 20],  # bool
        map_pos: [n_pl, 20, 3],  # float32
    Returns:
        map_boundary: [4]
    """
    pos = map_pos[map_valid]
    xmin = pos[:, 0].min()
    ymin = pos[:, 1].min()
    xmax = pos[:, 0].max()
    ymax = pos[:, 1].max()
    return np.array([xmin, xmax, ymin, ymax])


def classify_track(
    valid: np.ndarray,
    pos: np.ndarray,
    yaw: np.ndarray,
    spd: np.ndarray,
    kMaxSpeedForStationary: float = 2.0,  # (m/s)
    kMaxDisplacementForStationary: float = 5.0,  # (m)
    kMaxLateralDisplacementForStraight: float = 5.0,  # (m)
    kMinLongitudinalDisplacementForUTurn: float = -5.0,  # (m)
    kMaxAbsHeadingDiffForStraight: float = 0.5236,  # M_PI / 6.0
) -> int:
    """github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/motion_metrics_utils.cc
    Args:
        valid: [n_step], bool
        pos: [n_step, 2], x,y
        yaw: [n_step], float32
        spd: [n_step], float32
    Returns:
        traj_type: int in range(N_AGENT_CMD)
            # STATIONARY = 0;
            # STRAIGHT = 1;
            # STRAIGHT_LEFT = 2;
            # STRAIGHT_RIGHT = 3;
            # LEFT_U_TURN = 4;
            # LEFT_TURN = 5;
            # RIGHT_U_TURN = 6;
            # RIGHT_TURN = 7;
    """
    i0 = valid.argmax()
    i1 = len(valid) - 1 - np.flip(valid).argmax()

    x, y = pos[i1] - pos[i0]
    final_displacement = np.sqrt(x ** 2 + y ** 2)

    _c = np.cos(-yaw[i0])
    _s = np.sin(-yaw[i0])
    dx = x * _c - y * _s
    dy = x * _s + y * _c

    heading_diff = yaw[i1] - yaw[i0]
    max_speed = max(spd[i0], spd[i1])

    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return 0  # TrajectoryType::STATIONARY;

    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(dy) < kMaxLateralDisplacementForStraight:
            return 1  # TrajectoryType::STRAIGHT;
        if dy > 0:
            return 2  # TrajectoryType::STRAIGHT_LEFT
        else:
            return 3  # TrajectoryType::STRAIGHT_RIGHT

    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        if dx < kMinLongitudinalDisplacementForUTurn:
            return 6  # TrajectoryType::RIGHT_U_TURN
        else:
            return 7  # TrajectoryType::RIGHT_TURN

    if dx < kMinLongitudinalDisplacementForUTurn:
        return 4  # TrajectoryType::LEFT_U_TURN;

    return 5  # TrajectoryType::LEFT_TURN;


def pack_episode_map(
    episode: Dict[str, np.ndarray],
    mf_id: List[int],
    mf_xyz: List[List[List[float]]],
    mf_type: List[int],
    mf_edge: List[List[int]],
    n_pl_max: int,
    n_nodes: int = 20,
) -> int:
    """
    Args:
        mf_id: [polyline]
        mf_xyz: [polyline, points, xyz]
        mf_type: [polyline]
        mf_edge: [edge, 2]
    """
    episode["map/valid"] = np.zeros([n_pl_max, n_nodes], dtype=bool)
    episode["map/id"] = np.zeros([n_pl_max], dtype=np.int64) - 1
    episode["map/type"] = np.zeros([n_pl_max], dtype=np.int64)
    episode["map/pos"] = np.zeros([n_pl_max, n_nodes, 3], dtype=np.float32)
    episode["map/dir"] = np.zeros([n_pl_max, n_nodes, 3], dtype=np.float32)
    episode["map/edge"] = np.array(mf_edge)

    pl_counter = 0
    for i_pl in range(len(mf_id)):
        pl_pos = np.array(mf_xyz[i_pl])
        pl_dir = np.diff(pl_pos, axis=0)
        polyline_len = pl_dir.shape[0]
        polyline_cuts = np.linspace(0, polyline_len, polyline_len // n_nodes + 1, dtype=int, endpoint=False)
        num_cuts = len(polyline_cuts)
        for idx_cut in range(num_cuts):
            idx_start = polyline_cuts[idx_cut]
            if idx_cut + 1 == num_cuts:
                # last slice
                idx_end = polyline_len
            else:
                idx_end = polyline_cuts[idx_cut + 1]

            episode["map/valid"][pl_counter, : idx_end - idx_start] = True
            episode["map/pos"][pl_counter, : idx_end - idx_start] = pl_pos[idx_start:idx_end]
            episode["map/dir"][pl_counter, : idx_end - idx_start] = pl_dir[idx_start:idx_end]
            episode["map/type"][pl_counter] = mf_type[i_pl]
            episode["map/id"][pl_counter] = mf_id[i_pl]
            pl_counter += 1
    return pl_counter


def pack_episode_agents(
    episode: Dict[str, np.ndarray],
    agent_id: List[int],
    agent_type: List[int],
    agent_states: List[List[List[float]]],
    agent_role: List[List[bool]],
    pack_all: bool,
    pack_history: bool,
    n_agent_max: int,
    step_current: int,
    n_agent_type: int = 3,
) -> None:
    """
    Args:
        agent_id: [agents]
        agent_type: [agents]
        agent_states: [agents, step, 10]; x,y,z,l,w,h,heading,vx,vy,valid
        agent_role: [agents, :], bool, [sdc=0]
    """
    n_step = len(agent_states[0])
    data_agent_valid = np.zeros([n_step, n_agent_max], dtype=bool)
    data_agent_pos = np.zeros([n_step, n_agent_max, 3], dtype=np.float32)
    data_agent_vel = np.zeros([n_step, n_agent_max, 2], dtype=np.float32)
    data_agent_spd = np.zeros([n_step, n_agent_max, 1], dtype=np.float32)
    data_agent_yaw_bbox = np.zeros([n_step, n_agent_max, 1], dtype=np.float32)

    data_agent_type = np.zeros([n_agent_max, n_agent_type], dtype=bool)
    data_agent_cmd = np.zeros([n_agent_max, N_AGENT_CMD], dtype=bool)
    data_agent_role = np.zeros([n_agent_max, len(agent_role[0])], dtype=bool)
    data_agent_size = np.zeros([n_agent_max, 3], dtype=np.float32)
    data_agent_goal = np.zeros([n_agent_max, 4], dtype=np.float32)
    data_agent_object_id = np.zeros([n_agent_max], dtype=np.int64) - 1

    for i in range(len(agent_id)):
        data_agent_type[i, agent_type[i]] = True
        data_agent_object_id[i] = agent_id[i]
        data_agent_role[i] = agent_role[i]

        length = 0.0
        width = 0.0
        height = 0.0
        count = 0.0
        for k in range(n_step):
            if agent_states[i][k][9]:
                data_agent_pos[k, i, 0] = agent_states[i][k][0]
                data_agent_pos[k, i, 1] = agent_states[i][k][1]
                data_agent_pos[k, i, 2] = agent_states[i][k][2]
                length += agent_states[i][k][3]
                width += agent_states[i][k][4]
                height += agent_states[i][k][5]
                data_agent_yaw_bbox[k, i, 0] = agent_states[i][k][6]
                data_agent_vel[k, i, 0] = agent_states[i][k][7]
                data_agent_vel[k, i, 1] = agent_states[i][k][8]
                data_agent_valid[k, i] = agent_states[i][k][9]

                count += data_agent_valid[k, i]

                spd = np.linalg.norm(data_agent_vel[k, i], axis=-1)

                spd_sign = np.sign(
                    np.cos(data_agent_yaw_bbox[k, i, 0]) * data_agent_vel[k, i, 0]
                    + np.sin(data_agent_yaw_bbox[k, i, 0]) * data_agent_vel[k, i, 1]
                )
                data_agent_spd[k, i, 0] = spd * spd_sign

                # set goal as the last valid state [x,y,theta,v]
                data_agent_goal[i, 0] = data_agent_pos[k, i, 0]
                data_agent_goal[i, 1] = data_agent_pos[k, i, 1]
                data_agent_goal[i, 2] = data_agent_yaw_bbox[k, i, 0]
                data_agent_goal[i, 3] = data_agent_spd[k, i, 0]

        cmd = classify_track(
            data_agent_valid[step_current:, i],
            data_agent_pos[step_current:, i, :2],
            data_agent_yaw_bbox[step_current:, i, 0],
            data_agent_spd[step_current:, i, 0],
        )
        data_agent_cmd[i, cmd] = True

        data_agent_size[i, 0] = length / count if count > 0 else 0.0
        data_agent_size[i, 1] = width / count if count > 0 else 0.0
        data_agent_size[i, 2] = height / count if count > 0 else 0.0

    # swap sdc to be the first agent
    sdc_track_index = np.where(data_agent_role[:, 0])[0][0]
    data_agent_valid[:, [0, sdc_track_index]] = data_agent_valid[:, [sdc_track_index, 0]]
    data_agent_pos[:, [0, sdc_track_index]] = data_agent_pos[:, [sdc_track_index, 0]]
    data_agent_vel[:, [0, sdc_track_index]] = data_agent_vel[:, [sdc_track_index, 0]]
    data_agent_spd[:, [0, sdc_track_index]] = data_agent_spd[:, [sdc_track_index, 0]]
    data_agent_yaw_bbox[:, [0, sdc_track_index]] = data_agent_yaw_bbox[:, [sdc_track_index, 0]]
    data_agent_object_id[[0, sdc_track_index]] = data_agent_object_id[[sdc_track_index, 0]]
    data_agent_type[[0, sdc_track_index]] = data_agent_type[[sdc_track_index, 0]]
    data_agent_role[[0, sdc_track_index]] = data_agent_role[[sdc_track_index, 0]]
    data_agent_size[[0, sdc_track_index]] = data_agent_size[[sdc_track_index, 0]]
    data_agent_goal[[0, sdc_track_index]] = data_agent_goal[[sdc_track_index, 0]]

    if pack_all:
        episode["agent/valid"] = data_agent_valid.copy()
        episode["agent/pos"] = data_agent_pos.copy()
        episode["agent/vel"] = data_agent_vel.copy()
        episode["agent/spd"] = data_agent_spd.copy()
        episode["agent/yaw_bbox"] = data_agent_yaw_bbox.copy()

        episode["agent/object_id"] = data_agent_object_id.copy()
        episode["agent/type"] = data_agent_type.copy()
        episode["agent/role"] = data_agent_role.copy()
        episode["agent/size"] = data_agent_size.copy()
        episode["agent/cmd"] = data_agent_cmd.copy()
        episode["agent/goal"] = data_agent_goal.copy()
    if pack_history:
        episode["history/agent/valid"] = data_agent_valid[: step_current + 1].copy()
        episode["history/agent/pos"] = data_agent_pos[: step_current + 1].copy()
        episode["history/agent/vel"] = data_agent_vel[: step_current + 1].copy()
        episode["history/agent/spd"] = data_agent_spd[: step_current + 1].copy()
        episode["history/agent/yaw_bbox"] = data_agent_yaw_bbox[: step_current + 1].copy()

        episode["history/agent/object_id"] = data_agent_object_id.copy()
        episode["history/agent/type"] = data_agent_type.copy()
        episode["history/agent/role"] = data_agent_role.copy()
        episode["history/agent/size"] = data_agent_size.copy()
        # no goal, no cmd
        invalid = ~(episode["history/agent/valid"].any(0))
        episode["history/agent/object_id"][invalid] = -1
        episode["history/agent/type"][invalid] = False
        episode["history/agent/size"][invalid] = 0
    return len(agent_id)


def pack_episode_traffic_lights(
    episode: Dict[str, np.ndarray],
    tl_lane_state: List[List[float]],
    tl_lane_id: List[List[int]],
    tl_stop_point: List[List[List[float]]],
    pack_all: bool,
    pack_history: bool,
    n_tl_max: int,
    step_current: int,
) -> int:
    """
    Args:
        tl_lane_state: [step, tl_lane]
        tl_lane_id: [step, tl_lane]
        tl_stop_point: [step, tl_lane, 3], x,y,z
    """
    n_step = len(tl_lane_state)
    data_tl_lane_valid = np.zeros([n_step, n_tl_max], dtype=bool)
    data_tl_lane_state = np.zeros([n_step, n_tl_max], dtype=np.int64)
    data_tl_lane_id = np.zeros([n_step, n_tl_max], dtype=np.int64) - 1
    data_tl_stop_pos = np.zeros([n_step, n_tl_max, 3], dtype=np.float32)

    for _step in range(n_step):
        n_tl_lane = len(tl_lane_state[_step])
        if n_tl_lane > 0:
            data_tl_lane_valid[_step, :n_tl_lane] = True
            data_tl_lane_state[_step, :n_tl_lane] = np.array(tl_lane_state[_step])
            data_tl_lane_id[_step, :n_tl_lane] = np.array(tl_lane_id[_step])
            data_tl_stop_pos[_step, :n_tl_lane] = np.array(tl_stop_point[_step])

    if pack_all:
        episode["tl_lane/valid"] = data_tl_lane_valid.copy()
        episode["tl_lane/state"] = data_tl_lane_state.copy()
        episode["tl_lane/id"] = data_tl_lane_id.copy()
        episode["tl_stop/pos"] = data_tl_stop_pos.copy()
    if pack_history:
        episode["history/tl_lane/valid"] = data_tl_lane_valid[: step_current + 1].copy()
        episode["history/tl_lane/state"] = data_tl_lane_state[: step_current + 1].copy()
        episode["history/tl_lane/id"] = data_tl_lane_id[: step_current + 1].copy()
        episode["history/tl_stop/pos"] = data_tl_stop_pos[: step_current + 1].copy()
    return data_tl_lane_valid.sum(1).max()


def center_at_sdc(
    episode: Dict[str, np.ndarray], rand_pos: float = -1, rand_yaw: float = -1
) -> Tuple[np.ndarray, float]:
    """episode
    # agent state
    "agent/valid": [n_step, N_AGENT_MAX],  # bool,
    "agent/pos": [n_step, N_AGENT_MAX, 3],  # float32
    "agent/vel": [n_step, N_AGENT_MAX, 2],  # float32, v_x, v_y
    "agent/yaw_bbox": [n_step, N_AGENT_MAX, 1],  # float32, yaw of the bbox heading
    "agent/goal": [N_AGENT_MAX, 4],  # float32: [x, y, theta, v]
    # map
    "map/valid": [N_PL_MAX, n_pl_nodes],  # bool
    "map/pos": [N_PL_MAX, n_pl_nodes, 3],  # float32
    "map/dir": [N_PL_MAX, n_pl_nodes, 3],  # float32
    # traffic light
    "tl_lane/valid: [step, n_tl], bool
    "tl_stop/pos: [step, n_tl, 3], x,y,z
    """
    prefix = []
    if "agent/pos" in episode:
        prefix.append("")
    if "history/agent/valid" in episode:
        prefix.append("history/")

    sdc_center = episode[prefix[0] + "agent/pos"][0, 0, :2].copy()
    sdc_yaw = episode[prefix[0] + "agent/yaw_bbox"][0, 0, 0].copy()

    if rand_pos > 0:
        sdc_center[0] += np.random.uniform(-rand_pos, rand_pos)
        sdc_center[1] += np.random.uniform(-rand_pos, rand_pos)
    if rand_yaw > 0:
        sdc_yaw += np.random.uniform(-rand_yaw, rand_yaw)

    to_sdc_se3 = transform_utils.get_transformation_matrix(sdc_center, sdc_yaw)  # for points
    to_sdc_yaw = transform_utils.get_yaw_from_se2(to_sdc_se3)  # for vector
    to_sdc_so2 = transform_utils.get_so2_from_se2(to_sdc_se3)  # for angle

    # map
    episode["map/pos"][..., :2][episode["map/valid"]] = transform_utils.transform_points(
        episode["map/pos"][..., :2][episode["map/valid"]], to_sdc_se3
    )
    episode["map/dir"][..., :2][episode["map/valid"]] = transform_utils.transform_points(
        episode["map/dir"][..., :2][episode["map/valid"]], to_sdc_so2
    )

    for pf in prefix:
        # agent: pos, vel, yaw_bbox
        episode[pf + "agent/pos"][..., :2][episode[pf + "agent/valid"]] = transform_utils.transform_points(
            episode[pf + "agent/pos"][..., :2][episode[pf + "agent/valid"]], to_sdc_se3
        )
        episode[pf + "agent/vel"][episode[pf + "agent/valid"]] = transform_utils.transform_points(
            episode[pf + "agent/vel"][episode[pf + "agent/valid"]], to_sdc_so2
        )
        episode[pf + "agent/yaw_bbox"][episode[pf + "agent/valid"]] += to_sdc_yaw
        # traffic light: [step, tl, 3]
        key_tl = pf + "tl_stop/pos"
        if key_tl in episode:
            episode[key_tl][..., :2][episode[pf + "tl_lane/valid"]] = transform_utils.transform_points(
                episode[key_tl][..., :2][episode[pf + "tl_lane/valid"]], to_sdc_se3
            )
        if pf == "":
            # goal: x, y, theta
            goal_valid = episode["agent/valid"].any(axis=0)
            episode["agent/goal"][..., :2][goal_valid] = transform_utils.transform_points(
                episode["agent/goal"][..., :2][goal_valid], to_sdc_se3
            )
            episode["agent/goal"][..., 2][goal_valid] += to_sdc_yaw

    return sdc_center, sdc_yaw


def filter_episode_traffic_lights(episode: Dict[str, np.ndarray]) -> None:
    """Filter traffic light based on map valid
    Args: episode
        # map
        "map/valid": [N_PL_MAX, 20],  # bool
        "map/id": [N_PL_MAX],  # int, with -1
        "map/type": [N_PL_MAX],  # int, >= 0
        "map/pos": [N_PL_MAX, 20, 3]
        "map/dir": [N_PL_MAX, 20, 3]
        # traffic light
        "tl_lane/valid": [n_step, N_TL_MAX],  # bool
        "tl_lane/state": [n_step, N_TL_MAX],  # >= 0
        "tl_lane/id": [n_step, N_TL_MAX],  # with -1
        "tl_stop/pos": [n_step, N_TL_MAX, 3]
    """
    prefix = []
    if "tl_lane/valid" in episode:
        prefix.append("")
    if "history/tl_lane/valid" in episode:
        prefix.append("history/")
    for pf in prefix:
        n_step, n_tl_max = episode[pf + "tl_lane/valid"].shape
        for tl_step in range(n_step):
            for tl_idx in range(n_tl_max):
                if episode[pf + "tl_lane/valid"][tl_step, tl_idx]:
                    tl_lane_map_id = episode["map/id"] == episode[pf + "tl_lane/id"][tl_step, tl_idx]
                    if episode["map/valid"][tl_lane_map_id].sum() == 0:
                        episode[pf + "tl_lane/valid"][tl_step, tl_idx] = False


def filter_episode_map(episode: Dict[str, np.ndarray], n_pl: int, thresh_map: float, thresh_z: float = 3):
    """
    Args: episode
        "agent/valid": [n_step, N_AGENT_MAX], bool,
        "agent/pos": [n_step, N_AGENT_MAX, 3], float32
        "agent/role": [N_AGENT_MAX, :], bool, one hot
        "map/valid": [N_PL_MAX, n_pl_nodes], bool
        "map/id": [N_PL_MAX], int, with -1
        "map/type": [N_PL_MAX], int, >= 0
        "map/pos": [N_PL_MAX, n_pl_nodes, 3]
        "map/dir": [N_PL_MAX, n_pl_nodes, 3]
    """
    # ! filter "map/" based on distance to relevant agents, based on the history only
    if "agent/valid" in episode:
        relevant_agent = episode["agent/role"].any(-1)
        agent_valid_relevant = episode["agent/valid"][:11, relevant_agent]
        agent_pos_relevant = episode["agent/pos"][:11, relevant_agent]
    elif "history/agent/valid" in episode:
        relevant_agent = episode["history/agent/role"].any(-1)
        agent_valid_relevant = episode["history/agent/valid"][:, relevant_agent]
        agent_pos_relevant = episode["history/agent/pos"][:, relevant_agent]
    agent_pos_relevant = agent_pos_relevant[agent_valid_relevant]  # [N, 3]

    xmin = agent_pos_relevant[:, 0].min()
    xmax = agent_pos_relevant[:, 0].max()
    ymin = agent_pos_relevant[:, 1].min()
    ymax = agent_pos_relevant[:, 1].max()
    x_thresh = max(xmax - xmin, thresh_map)
    y_thresh = max(ymax - ymin, thresh_map)

    old_map_valid = episode["map/valid"].copy()
    episode["map/valid"] = episode["map/valid"] & (episode["map/pos"][..., 0] > xmin - x_thresh)
    episode["map/valid"] = episode["map/valid"] & (episode["map/pos"][..., 0] < xmax + x_thresh)
    episode["map/valid"] = episode["map/valid"] & (episode["map/pos"][..., 1] > ymin - y_thresh)
    episode["map/valid"] = episode["map/valid"] & (episode["map/pos"][..., 1] < ymax + y_thresh)
    if thresh_z > 0:
        zmin = agent_pos_relevant[:, 2].min()
        zmax = agent_pos_relevant[:, 2].max()
        z_thresh = max(zmax - zmin, thresh_z)
        episode["map/valid"] = episode["map/valid"] & (episode["map/pos"][..., 2] > zmin - z_thresh)
        episode["map/valid"] = episode["map/valid"] & (episode["map/pos"][..., 2] < zmax + z_thresh)

    # assert episode["map/valid"].sum() > 0
    if episode["map/valid"].any(1).sum() < 10:
        # in waymo training 83954, agent and map z axis is off, filter using zmin/zmax will remove all polylines.
        print("something is wrong, check this episode out!")
        episode["map/valid"] = old_map_valid

    # ! filter "map/" that has too little valid entries
    mask_map_short = episode["map/valid"].sum(1) <= 3
    episode["map/valid"][mask_map_short] = False

    # ! still too many polylines: filter polylines based on distance to relevant agents
    dist_thresh_map = thresh_map
    while episode["map/valid"].any(1).sum() > n_pl:
        mask_map_remain = episode["map/valid"].any(1)
        for i in range(episode["map/valid"].shape[0]):
            if mask_map_remain[i]:
                pl_pos = episode["map/pos"][i, :, :][episode["map/valid"][i, :]]
                close_to_agent = (
                    min(
                        np.linalg.norm(agent_pos_relevant - pl_pos[0], axis=1).min(),
                        np.linalg.norm(agent_pos_relevant - pl_pos[-1], axis=1).min(),
                    )
                    < dist_thresh_map
                )
                if not close_to_agent:
                    episode["map/valid"][i, :] = False
                if episode["map/valid"].any(1).sum() == n_pl:
                    break
        dist_thresh_map = dist_thresh_map * 0.5


def repack_episode_traffic_lights(
    episode: Dict[str, np.ndarray], episode_reduced: Dict[str, np.ndarray], n_tl: int, n_tl_state: int
) -> None:
    """
    Args: episode
        # map
        "map/valid": [N_PL_MAX, 20],  # bool
        "map/id": [N_PL_MAX],  # int, with -1
        "map/type": [N_PL_MAX],  # int, >= 0
        "map/pos": [N_PL_MAX, 20, 3]
        "map/dir": [N_PL_MAX, 20, 3]
        # traffic light
        "tl_lane/valid": [n_step, N_TL_MAX],  # bool
        "tl_lane/state": [n_step, N_TL_MAX],  # >= 0
        "tl_lane/id": [n_step, N_TL_MAX],  # with -1
        "tl_stop/pos": [n_step, N_TL_MAX, 3], # x,y,z
    Returns:
        "tl_lane/valid": [n_step, N_TL_MAX],  # bool
        "tl_lane/state": [n_step, N_TL_MAX],  # >= 0
        "tl_lane/id": [n_step, N_TL_MAX],  # with -1
        "tl_stop/valid": [n_step, N_TL_MAX],  # bool
        "tl_stop/pos": [n_step, N_TL_MAX, 3], # x,y,z
        "tl_stop/dir": [n_step, N_TL_MAX, 3], # x,y,z
        "tl_stop/state": [n_step, N_TL_MAX],  # >= 0
    """
    prefix = []
    if "tl_lane/valid" in episode:
        prefix.append("")
    if "history/tl_lane/valid" in episode:
        prefix.append("history/")
    for pf in prefix:
        n_step, n_tl_max = episode[pf + "tl_lane/valid"].shape
        # tl_lane
        episode_reduced[pf + "tl_lane/valid"] = np.zeros([n_step, n_tl], dtype=bool)  # bool
        episode_reduced[pf + "tl_lane/state"] = np.zeros([n_step, n_tl], dtype=np.int64)  # will be one_hot
        episode_reduced[pf + "tl_lane/idx"] = np.zeros([n_step, n_tl], dtype=np.int64) - 1  # int, -1 means not valid
        # tl_stop
        episode_reduced[pf + "tl_stop/valid"] = np.zeros([n_step, n_tl_max], dtype=bool)  # bool
        episode_reduced[pf + "tl_stop/state"] = np.zeros([n_step, n_tl_max], dtype=np.int64)  # will be one_hot
        episode_reduced[pf + "tl_stop/pos"] = np.zeros([n_step, n_tl_max, 2], dtype=np.float32)  # x,y
        episode_reduced[pf + "tl_stop/dir"] = np.zeros([n_step, n_tl_max, 2], dtype=np.float32)  # x,y
        for i in range(n_step):
            counter_tl = 0
            counter_tl_stop = 0
            for j in range(n_tl_max):
                if episode[pf + "tl_lane/valid"][i, j]:
                    lane_idx = np.where(episode_reduced["map/id"] == episode[pf + "tl_lane/id"][i, j])[0]
                    n_lanes = lane_idx.shape[0]
                    # assert counter_tl + n_lanes <= N_TL, print("counter_tl, n_lanes:", counter_tl, n_lanes)
                    episode_reduced[pf + "tl_lane/valid"][i, counter_tl : counter_tl + n_lanes] = True
                    episode_reduced[pf + "tl_lane/state"][i, counter_tl : counter_tl + n_lanes] = episode[
                        pf + "tl_lane/state"
                    ][i, j]
                    episode_reduced[pf + "tl_lane/idx"][i, counter_tl : counter_tl + n_lanes] = lane_idx
                    counter_tl += n_lanes
                    # tl_stop
                    episode_reduced[pf + "tl_stop/valid"][i, counter_tl_stop] = True
                    episode_reduced[pf + "tl_stop/state"][i, counter_tl_stop] = episode[pf + "tl_lane/state"][i, j]
                    episode_reduced[pf + "tl_stop/pos"][i, counter_tl_stop] = episode[pf + "tl_stop/pos"][i, j, :2]
                    episode_reduced[pf + "tl_stop/dir"][i, counter_tl_stop] = episode_reduced["map/dir"][
                        lane_idx[0], 0, :2
                    ]
                    counter_tl_stop += 1

        # one_hot "tl_lane/state" and "tl_stop/state": [N_AGENT, N_AGENT_TYPE], bool
        episode_reduced[pf + "tl_lane/state"] = np.eye(n_tl_state, dtype=bool)[episode_reduced[pf + "tl_lane/state"]]
        episode_reduced[pf + "tl_stop/state"] = np.eye(n_tl_state, dtype=bool)[episode_reduced[pf + "tl_stop/state"]]

        episode_reduced[pf + "tl_lane/state"] = (
            episode_reduced[pf + "tl_lane/state"] & episode_reduced[pf + "tl_lane/valid"][:, :, None]
        )
        episode_reduced[pf + "tl_stop/state"] = (
            episode_reduced[pf + "tl_stop/state"] & episode_reduced[pf + "tl_stop/valid"][:, :, None]
        )


def repack_episode_map(
    episode: Dict[str, np.ndarray], episode_reduced: Dict[str, np.ndarray], n_pl: int, n_pl_type: int
) -> int:
    """
    Args: episode
        # map
        "map/valid": [N_PL_MAX, 20],  # bool
        "map/id": [N_PL_MAX],  # int, with -1
        "map/type": [N_PL_MAX],  # int, >= 0
        "map/pos": [N_PL_MAX, 20, 3]
        "map/dir": [N_PL_MAX, 20, 3]
    """
    n_pl_nodes = episode["map/valid"].shape[1]
    episode_reduced["map/valid"] = np.zeros([n_pl, n_pl_nodes], dtype=bool)  # bool
    episode_reduced["map/type"] = np.zeros([n_pl], dtype=np.int64)  # will be one_hot
    episode_reduced["map/pos"] = np.zeros([n_pl, n_pl_nodes, 2], dtype=np.float32)  # x,y
    episode_reduced["map/dir"] = np.zeros([n_pl, n_pl_nodes, 2], dtype=np.float32)  # x,y
    episode_reduced["map/id"] = np.zeros([n_pl], dtype=np.int64) - 1

    map_valid_mask = episode["map/valid"].any(1)
    n_pl_valid = map_valid_mask.sum()
    episode_reduced["map/valid"][:n_pl_valid] = episode["map/valid"][map_valid_mask]
    episode_reduced["map/type"][:n_pl_valid] = episode["map/type"][map_valid_mask]
    episode_reduced["map/pos"][:n_pl_valid] = episode["map/pos"][map_valid_mask, :, :2]
    episode_reduced["map/dir"][:n_pl_valid] = episode["map/dir"][map_valid_mask, :, :2]
    episode_reduced["map/id"][:n_pl_valid] = episode["map/id"][map_valid_mask]
    # one_hot "map/type": [N_PL, N_PL_TYPE], bool
    episode_reduced["map/type"] = np.eye(n_pl_type, dtype=bool)[episode_reduced["map/type"]]
    episode_reduced["map/type"] = episode_reduced["map/type"] & episode_reduced["map/valid"].any(-1, keepdims=True)


def repack_episode_agents_no_sim(
    episode: Dict[str, np.ndarray],
    episode_reduced: Dict[str, np.ndarray],
    mask_no_sim: np.ndarray,
    n_agent_no_sim: int,
    prefix: str,
) -> None:
    n_step = episode[prefix + "agent/valid"].shape[0]
    episode_reduced[prefix + "agent_no_sim/valid"] = np.zeros([n_step, n_agent_no_sim], dtype=bool)
    episode_reduced[prefix + "agent_no_sim/pos"] = np.zeros([n_step, n_agent_no_sim, 2], dtype=np.float32)
    episode_reduced[prefix + "agent_no_sim/vel"] = np.zeros([n_step, n_agent_no_sim, 2], dtype=np.float32)
    episode_reduced[prefix + "agent_no_sim/spd"] = np.zeros([n_step, n_agent_no_sim, 1], dtype=np.float32)
    episode_reduced[prefix + "agent_no_sim/yaw_bbox"] = np.zeros([n_step, n_agent_no_sim, 1], dtype=np.float32)
    episode_reduced[prefix + "agent_no_sim/object_id"] = np.zeros([n_agent_no_sim], dtype=np.int64) - 1
    episode_reduced[prefix + "agent_no_sim/type"] = np.zeros([n_agent_no_sim, 3], dtype=bool)
    episode_reduced[prefix + "agent_no_sim/size"] = np.zeros([n_agent_no_sim, 3], dtype=np.float32)
    # no role, no cmd, no goal
    for i, idx in enumerate(np.where(mask_no_sim)[0]):
        episode_reduced[prefix + "agent_no_sim/valid"][:, i] = episode[prefix + "agent/valid"][:, idx]
        episode_reduced[prefix + "agent_no_sim/pos"][:, i] = episode[prefix + "agent/pos"][:, idx, :2]
        episode_reduced[prefix + "agent_no_sim/vel"][:, i] = episode[prefix + "agent/vel"][:, idx]
        episode_reduced[prefix + "agent_no_sim/spd"][:, i] = episode[prefix + "agent/spd"][:, idx]
        episode_reduced[prefix + "agent_no_sim/yaw_bbox"][:, i] = episode[prefix + "agent/yaw_bbox"][:, idx]
        episode_reduced[prefix + "agent_no_sim/object_id"][i] = episode[prefix + "agent/object_id"][idx]
        episode_reduced[prefix + "agent_no_sim/type"][i] = episode[prefix + "agent/type"][idx]
        episode_reduced[prefix + "agent_no_sim/size"][i] = episode[prefix + "agent/size"][idx]


def repack_episode_agents(
    episode: Dict[str, np.ndarray],
    episode_reduced: Dict[str, np.ndarray],
    mask_sim: np.ndarray,
    n_agent: int,
    prefix: str,
    dim_veh_lanes: List[int] = [],
    dim_cyc_lanes: List[int] = [],
    dim_ped_lanes: List[int] = [],
    dest_no_pred: bool = False,
) -> None:
    """fill episode_reduced["history/agent/"], episode_reduced["agent/"]
    Args: episode
        # agent state
        "agent/valid": [n_step, N_AGENT_MAX], bool,
        "agent/pos": [n_step, N_AGENT_MAX, 3], float32
        "agent/vel": [n_step, N_AGENT_MAX, 2], float32, v_x, v_y
        "agent/spd": [n_step, N_AGENT_MAX, 1], float32
        "agent/yaw_bbox": [n_step, N_AGENT_MAX, 1], float32, yaw of the bbox heading
        # agent attribute
        "agent/type": [N_AGENT_MAX, 3], bool, one hot [Vehicle=0, Pedestrian=1, Cyclist=2]
        "agent/cmd": [N_AGENT_MAX, N_AGENT_CMD], bool, one hot
        "agent/role": [N_AGENT_MAX, :], bool, one hot
        "agent/size": [N_AGENT_MAX, 3], float32: [length, width, height]
        "agent/goal": [N_AGENT_MAX, 4], float32: [x, y, theta, v]
        # map
        "map/valid": [N_PL_MAX, n_pl_nodes], bool
        "map/id": [N_PL_MAX], int, with -1
        "map/type": [N_PL_MAX], int, >= 0
        "map/pos": [N_PL_MAX, n_pl_nodes, 3]
        "map/dir": [N_PL_MAX, n_pl_nodes, 3]
    Returns: episode_reduced
    """
    n_step = episode[prefix + "agent/valid"].shape[0]
    # agent state
    episode_reduced[prefix + "agent/valid"] = np.zeros([n_step, n_agent], dtype=bool)  # bool,
    episode_reduced[prefix + "agent/pos"] = np.zeros([n_step, n_agent, 2], dtype=np.float32)  # x,y
    episode_reduced[prefix + "agent/vel"] = np.zeros([n_step, n_agent, 2], dtype=np.float32)  # v_x, v_y, in m/s
    episode_reduced[prefix + "agent/spd"] = np.zeros([n_step, n_agent, 1], dtype=np.float32)  # m/s, signed
    # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
    episode_reduced[prefix + "agent/acc"] = np.zeros([n_step, n_agent, 1], dtype=np.float32)
    episode_reduced[prefix + "agent/yaw_bbox"] = np.zeros([n_step, n_agent, 1], dtype=np.float32)  # [-pi, pi]
    # yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
    episode_reduced[prefix + "agent/yaw_rate"] = np.zeros([n_step, n_agent, 1], dtype=np.float32)
    # agent attribute
    episode_reduced[prefix + "agent/object_id"] = np.zeros([n_agent], dtype=np.int64) - 1
    episode_reduced[prefix + "agent/type"] = np.zeros([n_agent, 3], dtype=bool)
    # one hot [sdc=0, interest=1, predict=2]
    episode_reduced[prefix + "agent/role"] = np.zeros([n_agent, episode[prefix + "agent/role"].shape[-1]], dtype=bool)
    episode_reduced[prefix + "agent/size"] = np.zeros([n_agent, 3], dtype=np.float32)  # float32 [length, width, height]

    if prefix == "":
        episode_reduced["agent/cmd"] = np.zeros([n_agent, N_AGENT_CMD], dtype=bool)
        episode_reduced["agent/goal"] = np.zeros([n_agent, 4], dtype=np.float32)  # float32 [x, y, theta, v]
        episode_reduced["agent/dest"] = np.zeros([n_agent], dtype=np.int64)  # index to "map/valid" in [0, N_PL]
        # ! map info for finding dest
        n_pl, n_pl_node = episode_reduced["map/valid"].shape
        mask_veh_lane = (episode_reduced["map/type"][:, dim_veh_lanes].any(axis=-1, keepdims=True)) & episode_reduced[
            "map/valid"
        ]
        pos_veh_lane = episode_reduced["map/pos"][mask_veh_lane]  # [?, 2]
        dir_veh_lane = episode_reduced["map/dir"][mask_veh_lane]  # [?, 2]
        dir_veh_lane = dir_veh_lane / np.linalg.norm(dir_veh_lane, axis=-1, keepdims=True)
        map_id_veh_lane = episode_reduced["map/id"][:, None].repeat(n_pl_node, 1)[mask_veh_lane]
        pl_idx_veh_lane = np.arange(n_pl)[:, None].repeat(n_pl_node, 1)[mask_veh_lane]
        # cyc_lane
        mask_cyc_lane = (episode_reduced["map/type"][:, dim_cyc_lanes].any(axis=-1, keepdims=True)) & episode_reduced[
            "map/valid"
        ]
        pos_cyc_lane = episode_reduced["map/pos"][mask_cyc_lane]  # [?, 2]
        dir_cyc_lane = episode_reduced["map/dir"][mask_cyc_lane]  # [?, 2]
        dir_cyc_lane = dir_cyc_lane / np.linalg.norm(dir_cyc_lane, axis=-1, keepdims=True)
        pl_idx_cyc_lane = np.arange(n_pl)[:, None].repeat(n_pl_node, 1)[mask_cyc_lane]
        # road_edge
        mask_road_edge = (episode_reduced["map/type"][:, dim_ped_lanes].any(axis=-1, keepdims=True)) & episode_reduced[
            "map/valid"
        ]
        pos_road_edge = episode_reduced["map/pos"][mask_road_edge]  # [?, 2]
        pl_idx_road_edge = np.arange(n_pl)[:, None].repeat(n_pl_node, 1)[mask_road_edge]

    for i, idx in enumerate(np.where(mask_sim)[0]):
        valid = episode[prefix + "agent/valid"][:, idx]
        if valid.sum() > 1:
            valid_steps = np.where(valid)[0]
            step_start = valid_steps[0]
            step_end = valid_steps[-1]

            f_pos = interp1d(valid_steps, episode[prefix + "agent/pos"][valid, idx, :2], axis=0)
            f_vel = interp1d(valid_steps, episode[prefix + "agent/vel"][valid, idx], axis=0)
            f_spd = interp1d(valid_steps, episode[prefix + "agent/spd"][valid, idx], axis=0)
            f_yaw_bbox = interp1d(
                valid_steps, np.unwrap(episode[prefix + "agent/yaw_bbox"][valid, idx], axis=0), axis=0
            )

            x = np.arange(step_start, step_end + 1)
            x_spd = f_spd(x)
            x_yaw = f_yaw_bbox(x)
            episode_reduced[prefix + "agent/valid"][step_start : step_end + 1, i] = True
            episode_reduced[prefix + "agent/pos"][step_start : step_end + 1, i] = f_pos(x)
            episode_reduced[prefix + "agent/vel"][step_start : step_end + 1, i] = f_vel(x)
            episode_reduced[prefix + "agent/spd"][step_start : step_end + 1, i] = x_spd
            episode_reduced[prefix + "agent/yaw_bbox"][step_start : step_end + 1, i] = x_yaw

            x_acc = np.diff(x_spd, axis=0) / 0.1
            episode_reduced[prefix + "agent/acc"][step_start + 1 : step_end + 1, i] = x_acc
            x_yaw_rate = np.diff(x_yaw, axis=0) / 0.1
            episode_reduced[prefix + "agent/yaw_rate"][step_start + 1 : step_end + 1, i] = x_yaw_rate

        else:
            valid_step = np.where(valid)[0][0]
            episode_reduced[prefix + "agent/valid"][valid_step, i] = True
            episode_reduced[prefix + "agent/pos"][valid_step, i] = episode[prefix + "agent/pos"][valid_step, idx, :2]
            episode_reduced[prefix + "agent/vel"][valid_step, i] = episode[prefix + "agent/vel"][valid_step, idx]
            episode_reduced[prefix + "agent/spd"][valid_step, i] = episode[prefix + "agent/spd"][valid_step, idx]
            episode_reduced[prefix + "agent/yaw_bbox"][valid_step, i] = episode[prefix + "agent/yaw_bbox"][
                valid_step, idx
            ]

        episode_reduced[prefix + "agent/object_id"][i] = episode[prefix + "agent/object_id"][idx]
        episode_reduced[prefix + "agent/type"][i] = episode[prefix + "agent/type"][idx]
        episode_reduced[prefix + "agent/role"][i] = episode[prefix + "agent/role"][idx]
        episode_reduced[prefix + "agent/size"][i] = episode[prefix + "agent/size"][idx]

        if prefix == "":
            episode_reduced["agent/goal"][i] = episode["agent/goal"][idx]
            episode_reduced["agent/cmd"][i] = episode["agent/cmd"][idx]
            episode_reduced["agent/dest"][i] = find_dest(
                episode_reduced["agent/type"][i],
                episode_reduced["agent/goal"][i],
                episode["map/edge"],
                pos_veh_lane,
                dir_veh_lane,
                map_id_veh_lane,
                pl_idx_veh_lane,
                pos_cyc_lane,
                dir_cyc_lane,
                pl_idx_cyc_lane,
                pos_road_edge,
                pl_idx_road_edge,
                dest_no_pred,
            )


def find_dest(
    agent_type,  # one_hot [3]
    agent_goal,  # [x, y, theta, v]
    map_edge,  # [?, 2] id0 -> id1, or id0 -> -1
    pos_veh_lane,  # [?, 2]
    dir_veh_lane,  # [?, 2]
    map_id_veh_lane,  # [?]
    pl_idx_veh_lane,  # [?]
    pos_cyc_lane,  # [?, 2]
    dir_cyc_lane,  # [?, 2]
    pl_idx_cyc_lane,  # [?, 2]
    pos_road_edge,  # [?, 2]
    pl_idx_road_edge,  # [?]
    no_pred,
):
    goal_yaw = agent_goal[2]
    goal_heading = np.array([np.cos(goal_yaw), np.sin(goal_yaw)])
    goal_pos = agent_goal[:2]
    if no_pred:
        extended_goal_pos = goal_pos
    else:
        extended_goal_pos = agent_goal[:2] + goal_heading * agent_goal[3] * 5  # 5 seconds with constant speed
    if agent_type[0]:  # veh
        dist_pos = np.linalg.norm((pos_veh_lane - goal_pos), axis=1)
        dist_rot = np.dot(dir_veh_lane, goal_heading)
        candidate_lanes = (dist_pos < 3) & (dist_rot > 0)
        if candidate_lanes.any():  # associate to a lane, extend with map topology
            if no_pred:
                idx_dest = pl_idx_veh_lane[candidate_lanes][np.argmin(dist_pos[candidate_lanes])]
            else:
                dest_map_id = map_id_veh_lane[candidate_lanes][np.argmin(dist_pos[candidate_lanes])]
                next_map_id = dest_map_id
                counter = 0
                continue_extend = True
                while continue_extend:
                    next_edges = np.where(map_edge[:, 0] == next_map_id)[0]
                    dest_map_id, next_map_id = map_edge[np.random.choice(next_edges)]
                    counter += 1
                    # if (
                    #     (next_map_id not in map_id_veh_lane)
                    #     or ((len(next_edges) > 1) and (counter > 3))
                    #     or (counter > 6)
                    # ):
                    if (
                        (next_map_id not in map_id_veh_lane)
                        or ((len(next_edges) > 1) and (counter > 1))
                        or (counter > 3)
                    ):
                        continue_extend = False
                idx_dest = pl_idx_veh_lane[np.where(map_id_veh_lane == dest_map_id)[0][-1]]
        else:  # not associate to a lane, use road edge
            idx_dest = pl_idx_road_edge[np.linalg.norm((pos_road_edge - extended_goal_pos), axis=1).argmin()]
    elif agent_type[1]:  # ped
        idx_dest = pl_idx_road_edge[np.linalg.norm((pos_road_edge - extended_goal_pos), axis=1).argmin()]
    elif agent_type[2]:  # cyc
        dist_pos = np.linalg.norm((pos_cyc_lane - extended_goal_pos), axis=1)
        dist_rot = np.dot(dir_cyc_lane, goal_heading)
        candidate_lanes = (dist_pos < 3) & (dist_rot > 0)
        if candidate_lanes.any():  # associate to a bike lane, extend with constant vel and find bike lane
            idx_dest = pl_idx_cyc_lane[candidate_lanes][np.argmin(dist_pos[candidate_lanes])]
        else:  # not associate to a lane, use road edge
            idx_dest = pl_idx_road_edge[np.linalg.norm((pos_road_edge - extended_goal_pos), axis=1).argmin()]
    return idx_dest


def filter_episode_agents(
    episode: Dict[str, np.ndarray],
    episode_reduced: Dict[str, np.ndarray],
    n_agent: int,
    prefix: str,
    dim_veh_lanes: List[int],
    dist_thresh_agent: float,
    step_current: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Args: episode
        # agent state
        "agent/valid": [n_step, N_AGENT_MAX], bool,
        "agent/pos": [n_step, N_AGENT_MAX, 3], float32
        "agent/vel": [n_step, N_AGENT_MAX, 2], float32, v_x, v_y
        "agent/spd": [n_step, N_AGENT_MAX, 1], float32
        "agent/yaw_bbox": [n_step, N_AGENT_MAX, 1], float32, yaw of the bbox heading
        # agent attribute
        "agent/type": [N_AGENT_MAX, 3], bool, one hot [Vehicle=0, Pedestrian=1, Cyclist=2]
        "agent/cmd": [N_AGENT_MAX, N_AGENT_CMD], bool, one hot
        "agent/role": [N_AGENT_MAX, :], bool, one hot
        "agent/size": [N_AGENT_MAX, 3], float32: [length, width, height]
        "agent/goal": [N_AGENT_MAX, 4], float32: [x, y, theta, v]
        # map
        "map/valid": [N_PL_MAX, 20], bool
        "map/id": [N_PL_MAX], int, with -1
        "map/type": [N_PL_MAX], int, >= 0
        "map/pos": [N_PL_MAX, 20, 3]
        "map/dir": [N_PL_MAX, 20, 3]
    """
    n_agent_max = episode[prefix + "agent/valid"].shape[1]
    agent_valid = episode[prefix + "agent/valid"].copy()
    relevant_agent = episode[prefix + "agent/role"].any(-1)
    agent_valid_relevant = agent_valid[:, relevant_agent]
    agent_pos_relevant = episode[prefix + "agent/pos"][:, relevant_agent]
    agent_pos_relevant = agent_pos_relevant[agent_valid_relevant][:, :2]  # [N, 2]
    thresh_spd = 2 if prefix == "" else 0.5

    # ! filter agents not seen in the history.
    mask_agent_history_not_seen = (~relevant_agent) & ~(agent_valid[: step_current + 1].any(axis=0))
    agent_valid = agent_valid & ~mask_agent_history_not_seen[None, :]

    # ! filter agents that have small displacement and large distance to relevant agents or map polylines
    mask_agent_still = (
        (episode[prefix + "agent/spd"][..., 0].sum(axis=0) * 0.1 < thresh_spd)
        & (~relevant_agent)
        & (agent_valid.any(axis=0))
    )
    lane_pos = episode_reduced["map/pos"][episode_reduced["map/valid"]]
    for i in range(n_agent_max):
        # if mask_agent_still[i]:
        if mask_agent_still[i] and (agent_valid.any(axis=0).sum() > n_agent):
            agent_poses = episode[prefix + "agent/pos"][:, i, :2][agent_valid[:, i]]
            start_pos = agent_poses[0]
            end_pos = agent_poses[-1]
            far_to_relevant_agent = (np.linalg.norm(agent_pos_relevant - start_pos, axis=1).min() > 20) & (
                np.linalg.norm(agent_pos_relevant - end_pos, axis=1).min() > 20
            )
            far_to_lane = (np.linalg.norm(lane_pos - start_pos, axis=1).min() > 20) & (
                np.linalg.norm(lane_pos - end_pos, axis=1).min() > 20
            )
            if far_to_relevant_agent & far_to_lane:
                agent_valid[:, i] = False

    # ! filter parking vehicles that have large distance to relevant agents and lanes
    mask_veh_lane = (episode_reduced["map/type"][:, dim_veh_lanes].any(axis=-1, keepdims=True)) & episode_reduced[
        "map/valid"
    ]
    pos_veh_lane = episode_reduced["map/pos"][mask_veh_lane]  # [?, 2]
    dir_veh_lane = episode_reduced["map/dir"][mask_veh_lane]  # [?, 2]
    dir_veh_lane = dir_veh_lane / np.linalg.norm(dir_veh_lane, axis=-1, keepdims=True)

    mask_vehicle_still = (
        (episode[prefix + "agent/spd"][..., 0].sum(axis=0) * 0.1 < thresh_spd)
        & (~relevant_agent)
        & (agent_valid.any(axis=0))
        & (episode[prefix + "agent/type"][:, 0])
    )
    for i in range(n_agent_max):
        if mask_vehicle_still[i] and (agent_valid.any(axis=0).sum() > n_agent):
            agent_pos = episode[prefix + "agent/pos"][:, i, :2][agent_valid[:, i]][-1]
            agent_yaw = episode[prefix + "agent/yaw_bbox"][:, i, 0][agent_valid[:, i]][-1]
            agent_heading = np.array([np.cos(agent_yaw), np.sin(agent_yaw)])

            dist_pos = np.linalg.norm((pos_veh_lane - agent_pos), axis=1)
            dist_rot = np.dot(dir_veh_lane, agent_heading)
            candidate_lanes = (dist_pos < 3) & (dist_rot > 0)
            not_associate_to_lane = ~(candidate_lanes.any())

            far_to_relevant_agent = (np.linalg.norm(agent_pos_relevant - start_pos, axis=1).min() > 10) & (
                np.linalg.norm(agent_pos_relevant - end_pos, axis=1).min() > 10
            )
            if far_to_relevant_agent & not_associate_to_lane:
                agent_valid[:, i] = False

    # ! filter vehicles that have small displacement and large yaw change. Training only.
    if prefix == "" and (agent_valid.any(axis=0).sum() > n_agent):
        yaw_diff = np.abs(transform_utils.cast_rad(np.diff(episode["agent/yaw_bbox"][..., 0], axis=0))) * (
            agent_valid[:-1] & agent_valid[1:]
        )
        max_yaw_diff = yaw_diff.max(axis=0)
        mask_large_yaw_diff_veh = ((episode["agent/spd"][..., 0].sum(axis=0) * 0.1 < 6) & (max_yaw_diff > 0.5)) | (
            max_yaw_diff > 1.5
        )
        mask_large_yaw_diff_veh = mask_large_yaw_diff_veh & (episode["agent/type"][:, 0])
        mask_large_yaw_diff_ped_cyc = ((episode["agent/spd"][..., 0].sum(axis=0) * 0.1 < 1) & (max_yaw_diff > 0.5)) | (
            max_yaw_diff > 1.5
        )
        mask_large_yaw_diff_ped_cyc = mask_large_yaw_diff_ped_cyc & (episode["agent/type"][:, 1:].any(-1))
        mask_agent_large_yaw_change = (
            (mask_large_yaw_diff_veh | mask_large_yaw_diff_ped_cyc) & (~relevant_agent) & (agent_valid.any(axis=0))
        )
        agent_valid[:, mask_agent_large_yaw_change] = False

    # ! still too many agents: filter agents based on distance to relevant agents
    while agent_valid.any(axis=0).sum() > n_agent:
        mask_agent_remain = ~(relevant_agent) & (agent_valid.any(axis=0))
        for i in range(n_agent_max):
            if mask_agent_remain[i]:
                agent_poses = episode[prefix + "agent/pos"][:, i, :2][agent_valid[:, i]]
                close_to_relevant_agent = (
                    min(
                        np.linalg.norm(agent_pos_relevant - agent_poses[0], axis=1).min(),
                        np.linalg.norm(agent_pos_relevant - agent_poses[-1], axis=1).min(),
                    )
                    < dist_thresh_agent
                )
                if not close_to_relevant_agent:
                    agent_valid[:, i] = False
                if agent_valid.any(axis=0).sum() == n_agent:
                    break
        dist_thresh_agent = dist_thresh_agent * 0.5

    mask_sim = agent_valid.any(0)
    mask_no_sim = episode[prefix + "agent/valid"].any(0) & (~mask_sim)
    return mask_sim, mask_no_sim
