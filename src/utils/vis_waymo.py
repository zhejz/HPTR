# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from torch import Tensor
import torch

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_VIOLET = (170, 0, 255)

COLOR_BUTTER_0 = (252, 233, 79)
COLOR_BUTTER_1 = (237, 212, 0)
COLOR_BUTTER_2 = (196, 160, 0)
COLOR_ORANGE_0 = (252, 175, 62)
COLOR_ORANGE_1 = (245, 121, 0)
COLOR_ORANGE_2 = (209, 92, 0)
COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHOCOLATE_1 = (193, 125, 17)
COLOR_CHOCOLATE_2 = (143, 89, 2)
COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_CHAMELEON_1 = (115, 210, 22)
COLOR_CHAMELEON_2 = (78, 154, 6)
COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (52, 101, 164)
COLOR_SKY_BLUE_2 = (32, 74, 135)
COLOR_PLUM_0 = (173, 127, 168)
COLOR_PLUM_1 = (117, 80, 123)
COLOR_PLUM_2 = (92, 53, 102)
COLOR_SCARLET_RED_0 = (239, 41, 41)
COLOR_SCARLET_RED_1 = (204, 0, 0)
COLOR_SCARLET_RED_2 = (164, 0, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (186, 189, 182)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_4 = (85, 87, 83)
COLOR_ALUMINIUM_4_5 = (66, 62, 64)
COLOR_ALUMINIUM_5 = (46, 52, 54)


class VisWaymo:
    def __init__(
        self,
        map_valid: np.ndarray,
        map_type: np.ndarray,
        map_pos: np.ndarray,
        map_boundary: np.ndarray,
        px_per_m: float = 10.0,
        video_size: int = 960,
    ) -> None:
        # centered around ego vehicle first step, x=0, y=0, theta=0
        self.px_per_m = px_per_m
        self.video_size = video_size
        self.px_agent2bottom = video_size // 2

        # waymo
        self.lane_style = [
            (COLOR_WHITE, 6),  # FREEWAY = 0
            (COLOR_ALUMINIUM_4_5, 6),  # SURFACE_STREET = 1
            (COLOR_ORANGE_2, 6),  # STOP_SIGN = 2
            (COLOR_CHOCOLATE_2, 6),  # BIKE_LANE = 3
            (COLOR_SKY_BLUE_2, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM_2, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_BUTTER_0, 2),  # BROKEN = 6
            (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
            (COLOR_SCARLET_RED_2, 2),  # DOUBLE = 8
            (COLOR_CHAMELEON_2, 4),  # SPEED_BUMP = 9
            (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        ]
        # argoverse
        # self.lane_style = [
        #     (COLOR_WHITE, 6),  # VEHICLE = 0
        #     (COLOR_ALUMINIUM_4_5, 6),  # BUS = 1
        #     (COLOR_ORANGE_2, 6),  # BIKE = 2
        #     (COLOR_CHOCOLATE_2, 6),  # UNKNOWN = 3
        #     (COLOR_SKY_BLUE_2, 4),  # DOUBLE_DASH = 4
        #     (COLOR_PLUM_2, 4),  # DASHED = 5
        #     (COLOR_BUTTER_0, 2),  # SOLID = 6
        #     (COLOR_MAGENTA, 2),  # DOUBLE_SOLID = 7
        #     (COLOR_SCARLET_RED_2, 2),  # DASH_SOLID = 8
        #     (COLOR_CHAMELEON_2, 4),  # SOLID_DASH = 9
        #     (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        # ]

        self.tl_style = [
            COLOR_ALUMINIUM_1,  # STATE_UNKNOWN = 0;
            COLOR_RED,  # STOP = 1;
            COLOR_YELLOW,  # CAUTION = 2;
            COLOR_GREEN,  # GO = 3;
            COLOR_VIOLET,  # FLASHING = 4;
        ]
        # sdc=0, interest=1, predict=2
        self.agent_role_style = [COLOR_CYAN, COLOR_CHAMELEON_2, COLOR_MAGENTA]

        self.agent_cmd_txt = [
            "STATIONARY",  # STATIONARY = 0;
            "STRAIGHT",  # STRAIGHT = 1;
            "STRAIGHT_LEFT",  # STRAIGHT_LEFT = 2;
            "STRAIGHT_RIGHT",  # STRAIGHT_RIGHT = 3;
            "LEFT_U_TURN",  # LEFT_U_TURN = 4;
            "LEFT_TURN",  # LEFT_TURN = 5;
            "RIGHT_U_TURN",  # RIGHT_U_TURN = 6;
            "RIGHT_TURN",  # RIGHT_TURN = 7;
        ]

        raster_map, self.top_left_px = self._register_map(map_boundary, self.px_per_m)
        self.raster_map = self._draw_map(raster_map, map_valid, map_type, map_pos)

    def save_prediction_videos(
        self, video_base_name: str, episode: Dict[str, np.ndarray], prediction: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, Tuple]:
        """
        Args:
            episode["agent/valid"]: np.zeros([n_step, N_AGENT], dtype=bool),  # bool,
            episode["agent/pos"]: np.zeros([n_step, N_AGENT, 2], dtype=np.float32),  # x,y
            episode["agent/yaw_bbox"]: np.zeros([n_step, N_AGENT, 1], dtype=np.float32),  # [-pi, pi]
            episode["agent/role"]: np.zeros([N_AGENT, 3], dtype=bool) one_hot [sdc=0, interest=1, predict=2]
            episode["agent/size"]: np.zeros([N_AGENT, 3], dtype=np.float32),  # float32: [length, width, height]
            episode["map/valid"]: np.zeros([N_PL, 20], dtype=bool),  # bool
            episode["map/pos"]: np.zeros([N_PL, 20, 2], dtype=np.float32),  # x,y
            episode["tl_lane/valid"]: np.zeros([n_step, N_TL], dtype=bool),  # bool
            episode["tl_lane/state"]: np.zeros([n_step, N_TL, N_TL_STATE], dtype=bool),  # one_hot
            episode["tl_lane/idx"]: np.zeros([n_step, N_TL], dtype=np.int64) - 1,  # int, -1 means not valid
            episode["tl_stop/valid"]: np.zeros([n_step, N_TL_STOP], dtype=bool),  # bool
            episode["tl_stop/state"]: np.zeros([n_step, N_TL_STOP, N_TL_STATE], dtype=bool),  # one_hot
            episode["tl_stop/pos"]: np.zeros([n_step, N_TL_STOP, 2], dtype=np.float32)  # x,y
            episode["tl_stop/dir"]: np.zeros([n_step, N_TL_STOP, 2], dtype=np.float32)  # x,y
            prediction["step_current"] <= prediction["step_gt"] <= prediction["step_end"]
            prediction["agent/valid"]: [step_current+1...step_end, N_AGENT]
            prediction["agent/pos"]: [step_current+1...step_end, N_AGENT, 2]
            prediction["agent/yaw_bbox"]: [step_current+1...step_end, N_AGENT, 1]
        """
        buffer_video = {f"{video_base_name}-gt.mp4": [[], None]}  # [List[im], agent_id]
        if prediction is None:
            step_end = episode["agent/valid"].shape[0] - 1
            step_gt = step_end
        else:
            step_end = prediction["step_end"]
            step_gt = prediction["step_gt"]
            buffer_video[f"{video_base_name}-pd.mp4"] = [[], None]
            buffer_video[f"{video_base_name}-mix.mp4"] = [[], None]

        for t in range(step_end + 1):
            step_image = self.raster_map.copy()
            # draw traffic lights
            if "tl_lane/valid" in episode:
                t_tl = min(t, step_gt)
                for i in range(episode["tl_lane/valid"].shape[1]):
                    if episode["tl_lane/valid"][t_tl, i]:
                        lane_idx = episode["tl_lane/idx"][t_tl, i]
                        tl_state = episode["tl_lane/state"][t_tl, i].argmax()
                        pos = self._to_pixel(episode["map/pos"][lane_idx][episode["map/valid"][lane_idx]])
                        cv2.polylines(
                            step_image,
                            [pos],
                            isClosed=False,
                            color=self.tl_style[tl_state],
                            thickness=8,
                            lineType=cv2.LINE_AA,
                        )
                        if tl_state >= 1 and tl_state <= 3:
                            cv2.drawMarker(
                                step_image,
                                pos[-1],
                                color=self.tl_style[tl_state],
                                markerType=cv2.MARKER_TILTED_CROSS,
                                markerSize=10,
                                thickness=6,
                            )
            # draw traffic lights stop points
            if "tl_stop/valid" in episode:
                for i in range(episode["tl_stop/valid"].shape[1]):
                    if episode["tl_stop/valid"][t_tl, i]:
                        tl_state = episode["tl_stop/state"][t_tl, i].argmax()
                        stop_point = self._to_pixel(episode["tl_stop/pos"][t_tl, i])
                        stop_point_end = self._to_pixel(
                            episode["tl_stop/pos"][t_tl, i] + 5 * episode["tl_stop/dir"][t_tl, i]
                        )
                        cv2.arrowedLine(
                            step_image,
                            stop_point,
                            stop_point_end,
                            color=self.tl_style[tl_state],
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.3,
                        )

            # draw agents: prediction["step_current"] <= prediction["step_gt"] <= prediction["step_end"]
            step_image_gt = step_image.copy()
            raster_blend_gt = np.zeros_like(step_image)
            if t <= step_gt:
                bbox_gt = self._get_agent_bbox(
                    episode["agent/valid"][t],
                    episode["agent/pos"][t],
                    episode["agent/yaw_bbox"][t],
                    episode["agent/size"],
                )
                bbox_gt = self._to_pixel(bbox_gt)
                agent_role = episode["agent/role"][episode["agent/valid"][t]]
                heading_start = self._to_pixel(episode["agent/pos"][t][episode["agent/valid"][t]])
                heading_end = self._to_pixel(
                    episode["agent/pos"][t][episode["agent/valid"][t]]
                    + 1.5
                    * np.stack(
                        [
                            np.cos(episode["agent/yaw_bbox"][t, :, 0][episode["agent/valid"][t]]),
                            np.sin(episode["agent/yaw_bbox"][t, :, 0][episode["agent/valid"][t]]),
                        ],
                        axis=-1,
                    )
                )
                for i in range(agent_role.shape[0]):
                    if not agent_role[i].any():
                        color = COLOR_ALUMINIUM_0
                    else:
                        color = self.agent_role_style[np.where(agent_role[i])[0].min()]
                    cv2.fillConvexPoly(step_image_gt, bbox_gt[i], color=color)
                    cv2.fillConvexPoly(raster_blend_gt, bbox_gt[i], color=color)
                    cv2.arrowedLine(
                        step_image_gt,
                        heading_start[i],
                        heading_end[i],
                        color=COLOR_BLACK,
                        thickness=4,
                        line_type=cv2.LINE_AA,
                        tipLength=0.6,
                    )
            buffer_video[f"{video_base_name}-gt.mp4"][0].append(step_image_gt)

            if prediction is not None:
                if t > prediction["step_current"]:
                    step_image_pd = step_image.copy()
                    t_pred = t - prediction["step_current"] - 1
                    bbox_pred = self._get_agent_bbox(
                        prediction["agent/valid"][t_pred],
                        prediction["agent/pos"][t_pred],
                        prediction["agent/yaw_bbox"][t_pred],
                        episode["agent/size"],
                    )
                    bbox_pred = self._to_pixel(bbox_pred)
                    heading_start = self._to_pixel(prediction["agent/pos"][t_pred][prediction["agent/valid"][t_pred]])
                    heading_end = self._to_pixel(
                        prediction["agent/pos"][t_pred][prediction["agent/valid"][t_pred]]
                        + 1.5
                        * np.stack(
                            [
                                np.cos(prediction["agent/yaw_bbox"][t_pred, :, 0][prediction["agent/valid"][t_pred]]),
                                np.sin(prediction["agent/yaw_bbox"][t_pred, :, 0][prediction["agent/valid"][t_pred]]),
                            ],
                            axis=-1,
                        )
                    )
                    agent_role = episode["agent/role"][prediction["agent/valid"][t_pred]]
                    for i in range(agent_role.shape[0]):
                        if not agent_role[i].any():
                            color = COLOR_ALUMINIUM_0
                        else:
                            color = self.agent_role_style[np.where(agent_role[i])[0].min()]
                        cv2.fillConvexPoly(step_image_pd, bbox_pred[i], color=color)
                        cv2.arrowedLine(
                            step_image_pd,
                            heading_start[i],
                            heading_end[i],
                            color=COLOR_BLACK,
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.6,
                        )
                    # step_image_mix = step_image.copy()
                    # cv2.addWeighted(raster_blend_gt, 0.6, step_image_pd, 1, 0, step_image_mix)
                    step_image_mix = cv2.addWeighted(raster_blend_gt, 0.6, step_image_pd, 1, 0)
                else:
                    step_image_pd = step_image_gt.copy()
                    step_image_mix = step_image_gt.copy()
                buffer_video[f"{video_base_name}-pd.mp4"][0].append(step_image_pd)
                buffer_video[f"{video_base_name}-mix.mp4"][0].append(step_image_mix)

        for k, v in buffer_video.items():
            encoder = ImageEncoder(k, v[0][0].shape, 20, 20)
            for im in v[0]:
                encoder.capture_frame(im)
            encoder.close()
            encoder = None
            buffer_video[k] = v[0][0].shape

        return buffer_video

    def save_prediction_images(
        self,
        im_path: str,
        step_current: int,
        gt_valid: np.ndarray,
        gt_pos: np.ndarray,
        gt_yaw: np.ndarray,
        gt_size: np.ndarray,
        pred_xy: np.ndarray,
        pred_scores: np.ndarray,
        cmd: int,
        episode: Dict[str, np.ndarray],
        vis_cmd: bool = False,
    ) -> Tensor:
        """
        Args:
            im_path: ".jpg"
            step_current: 10
            gt_valid: [91], bool
            gt_pos: [91, 2]
            gt_yaw: [91, 1]
            gt_size: [3] lwh
            pred_xy: [80, n_pred, 2]
            pred_scores: [n_pred]
        """
        image = self.raster_map.copy()
        if vis_cmd:
            image = cv2.putText(
                image, self.agent_cmd_txt[cmd], (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4
            )
        # ! draw traffic lights
        if "tl_lane/valid" in episode:
            t_tl = step_current
            for i in range(episode["tl_lane/valid"].shape[1]):
                if episode["tl_lane/valid"][t_tl, i]:
                    lane_idx = episode["tl_lane/idx"][t_tl, i]
                    tl_state = episode["tl_lane/state"][t_tl, i].argmax()
                    pos = self._to_pixel(episode["map/pos"][lane_idx][episode["map/valid"][lane_idx]])
                    cv2.polylines(
                        image,
                        [pos],
                        isClosed=False,
                        color=self.tl_style[tl_state],
                        thickness=8,
                        lineType=cv2.LINE_AA,
                    )
        # draw traffic lights stop points
        if "tl_stop/valid" in episode:
            for i in range(episode["tl_stop/valid"].shape[1]):
                if episode["tl_stop/valid"][t_tl, i]:
                    tl_state = episode["tl_stop/state"][t_tl, i].argmax()
                    stop_point = self._to_pixel(episode["tl_stop/pos"][t_tl, i])
                    stop_point_end = self._to_pixel(
                        episode["tl_stop/pos"][t_tl, i] + 5 * episode["tl_stop/dir"][t_tl, i]
                    )
                    cv2.arrowedLine(
                        image,
                        stop_point,
                        stop_point_end,
                        color=self.tl_style[tl_state],
                        thickness=4,
                        line_type=cv2.LINE_AA,
                        tipLength=0.3,
                    )
        # ! draw all agents
        bbox_gt = self._get_agent_bbox(
            episode["agent/valid"][step_current],
            episode["agent/pos"][step_current],
            episode["agent/yaw_bbox"][step_current],
            episode["agent/size"],
        )
        bbox_gt = self._to_pixel(bbox_gt)
        heading_start = self._to_pixel(episode["agent/pos"][step_current][episode["agent/valid"][step_current]])
        heading_end = self._to_pixel(
            episode["agent/pos"][step_current][episode["agent/valid"][step_current]]
            + 1.5
            * np.stack(
                [
                    np.cos(episode["agent/yaw_bbox"][step_current, :, 0][episode["agent/valid"][step_current]]),
                    np.sin(episode["agent/yaw_bbox"][step_current, :, 0][episode["agent/valid"][step_current]]),
                ],
                axis=-1,
            )
        )
        for i in range(bbox_gt.shape[0]):
            color = COLOR_WHITE
            cv2.fillConvexPoly(image, bbox_gt[i], color=color)
            cv2.arrowedLine(
                image,
                heading_start[i],
                heading_end[i],
                color=COLOR_BLACK,
                thickness=4,
                line_type=cv2.LINE_AA,
                tipLength=0.6,
            )

        # ! draw predictions
        pred_px = self._to_pixel(pred_xy)  # [80, n_pred, 2]
        v_channel = pred_scores / np.max(pred_scores) * 0.6 + 0.3  # pred_scores: [n_pred]
        for k_order, k in enumerate(pred_scores.argsort()):
            image_pred = np.zeros_like(image)
            color = COLOR_CYAN
            thickness = max(1, k_order - len(pred_scores) + 10)
            for i in range(pred_xy.shape[0] - 1):
                cv2.line(
                    image_pred, pred_px[i, k], pred_px[i + 1, k], color=color, thickness=thickness, lineType=cv2.LINE_AA
                )
            cv2.drawMarker(
                image_pred,
                pred_px[-1, k],
                color=color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=thickness * 3,
                thickness=thickness,
            )
            image = cv2.addWeighted(image_pred, v_channel[k], image, 1, 0)

        # ! draw gt
        gt_px = self._to_pixel(gt_pos)  # [91, 2]
        gt_last_valid = step_current
        for i in range(step_current, gt_valid.shape[0] - 1):
            if gt_valid[i] & gt_valid[i + 1]:
                color = COLOR_ORANGE_1
                thickness = 4
                cv2.line(image, gt_px[i], gt_px[i + 1], color=color, thickness=thickness, lineType=cv2.LINE_AA)
                gt_last_valid = i + 1
        cv2.drawMarker(
            image,
            gt_px[gt_last_valid],
            color=color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=thickness * 3,
            thickness=thickness,
        )

        # ! draw prediction agent
        k_draw = np.where(gt_valid[: step_current + 1])[0][-1]
        heading_start = self._to_pixel(gt_pos[k_draw, :2])
        heading_end = self._to_pixel(
            gt_pos[k_draw, :2] + 1.5 * np.concatenate([np.cos(gt_yaw[k_draw]), np.sin(gt_yaw[k_draw])], axis=-1)
        )

        bbox_hist = self._get_agent_bbox(gt_valid[k_draw], gt_pos[k_draw, :2], gt_yaw[k_draw], gt_size[:2])
        bbox_hist = self._to_pixel(bbox_hist)

        cv2.fillConvexPoly(image, bbox_hist, color=COLOR_CYAN)
        cv2.arrowedLine(
            image, heading_start, heading_end, color=COLOR_BLACK, thickness=4, line_type=cv2.LINE_AA, tipLength=0.6
        )

        cv2.imwrite(im_path, image[..., ::-1])
        return torch.from_numpy(image)

    def _draw_map(
        self, raster_map: np.ndarray, map_valid: np.ndarray, map_type: np.ndarray, map_pos: np.ndarray
    ) -> np.ndarray:
        """
        Args: numpy arrays
            map_valid: [n_pl, 20],  # bool
            map_type: [n_pl, 11],  # bool one_hot
            map_pos: [n_pl, 20, 2],  # float32

        Returns:
            raster_map
        """
        mask_valid = map_valid.any(axis=1)

        for type_to_draw in range(len(self.lane_style)):
            for i in np.where((map_type[:, type_to_draw]) & mask_valid)[0]:
                color, thickness = self.lane_style[type_to_draw]
                cv2.polylines(
                    raster_map,
                    [self._to_pixel(map_pos[i][map_valid[i]])],
                    isClosed=False,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        return raster_map

    @staticmethod
    def _register_map(map_boundary: np.ndarray, px_per_m: float, edge_px: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            map_boundary: [4], xmin, xmax, ymin, ymax
            px_per_m: float

        Returns:
            raster_map: empty image
            top_left_px
        """
        # y axis is inverted in pixel coordinate
        xmin, xmax, ymax, ymin = (map_boundary * px_per_m).astype(np.int64)
        ymax *= -1
        ymin *= -1
        xmin -= edge_px
        ymin -= edge_px
        xmax += edge_px
        ymax += edge_px

        raster_map = np.zeros([ymax - ymin, xmax - xmin, 3], dtype=np.uint8)
        top_left_px = np.array([xmin, ymin], dtype=np.float32)
        return raster_map, top_left_px

    @staticmethod
    def _get_agent_bbox(
        agent_valid: np.ndarray, agent_pos: np.ndarray, agent_yaw: np.ndarray, agent_size: np.ndarray
    ) -> np.ndarray:
        yaw = agent_yaw[agent_valid]  # n, 1
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        v_forward = np.concatenate([cos_yaw, sin_yaw], axis=-1)  # n,2
        v_right = np.concatenate([sin_yaw, -cos_yaw], axis=-1)

        offset_forward = 0.5 * agent_size[agent_valid, 0:1] * v_forward  # [n, 2]
        offset_right = 0.5 * agent_size[agent_valid, 1:2] * v_right  # [n, 2]

        vertex_offset = np.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            axis=1,
        )  # n,4,2

        agent_pos = agent_pos[agent_valid]
        bbox = agent_pos[:, None, :].repeat(4, 1) + vertex_offset  # n,4,2
        return bbox

    def _to_pixel(self, pos: np.ndarray) -> np.ndarray:
        pos = pos * self.px_per_m
        pos[..., 0] = pos[..., 0] - self.top_left_px[0]
        pos[..., 1] = -pos[..., 1] - self.top_left_px[1]
        return np.round(pos).astype(np.int32)
