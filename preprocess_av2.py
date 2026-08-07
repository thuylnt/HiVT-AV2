"""Preprocess Argoverse 2 -> HiVT TemporalData format.

Reads raw AV2 parquet (position + heading + 2D velocity + per-scenario map) and writes .pt files in the format HiVT needs, with two additions:
  - x_vel: 2D velocity measured directly (already rotated into the frame), as an experiment to replace HiVT's self-computed displacement.
  - rotate_angles uses the REAL AV2 heading (not the motion direction).

Actor/lane extraction logic is adapted from Forecast-MAE (jchengai/forecast-mae, av2_extractor), modified to (a) keep 2D velocity, (b) output TemporalData (graph) instead of a padded dict.

Run:  python preprocess_av2.py --data_root <data_av2> --split val [-p]
"""
import os
import argparse
from pathlib import Path
from itertools import permutations, product

import numpy as np
import torch

import av2.geometry.interpolate as interp_utils
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.lane_segment import LaneType
import pandas as pd

from utils import TemporalData

HIST, FUT, TOT = 50, 60, 110            # AV2: 5s history + 6s future @10Hz
LOCAL_RADIUS = 50.0                     # lane-actor radius (same as HiVT)
MAP_RADIUS = 100.0                      # radius for pulling lanes around the focal

OBJECT_TYPE = {
    ObjectType.VEHICLE.value: 0, ObjectType.PEDESTRIAN.value: 1,
    ObjectType.MOTORCYCLIST.value: 2, ObjectType.CYCLIST.value: 3,
    ObjectType.BUS.value: 4, ObjectType.STATIC.value: 5,
    ObjectType.BACKGROUND.value: 6, ObjectType.CONSTRUCTION.value: 7,
    ObjectType.RIDERLESS_BICYCLE.value: 8, ObjectType.UNKNOWN.value: 9,
}
IGNORE = {5, 6, 7, 8, 9}                # types that do not predict a future for
LANE_TYPE = {LaneType.VEHICLE.value: 0, LaneType.BIKE.value: 1, LaneType.BUS.value: 2}


def load_av2(scenario_file: Path):
    sid = scenario_file.stem.split("_")[-1]
    df = pd.read_parquet(scenario_file)
    am = ArgoverseStaticMap.from_json(scenario_file.parents[0] / f"log_map_archive_{sid}.json")
    return df, am, sid


def get_lanes(am, origin, rotate_mat):
    """Return the (rotated) centerline + intersection flag for each lane near the focal."""
    segs = am.get_nearby_lane_segments(origin.numpy(), MAP_RADIUS)
    cls, inters = [], []
    for s in segs:
        cl, _ = interp_utils.compute_midpoint_line(
            left_ln_boundary=s.left_lane_boundary.xyz,
            right_ln_boundary=s.right_lane_boundary.xyz, num_interp_pts=20)
        cl = torch.from_numpy(cl[:, :2]).float()
        cl = torch.matmul(cl - origin, rotate_mat)
        cls.append(cl)
        inters.append(float(am.lane_is_in_intersection(s.id)))
    return cls, inters                  # list[[20,2]], list[bool]

def process(scenario_file: Path, mode: str):
    df, am, sid = load_av2(scenario_file)
    city = df.city.values[0]
    focal = df["focal_track_id"].values[0]

    ts = list(np.sort(df["timestep"].unique()))
    cur = df[df["timestep"] == ts[49]]
    ids = list(cur["track_id"].unique())
    cur_pos = torch.from_numpy(cur[["position_x", "position_y"]].values).float()

    # origin = focal @ t=49, rotate by the focal's REAL heading
    fdf = df[df["track_id"] == focal].iloc
    origin = torch.tensor([fdf[49]["position_x"], fdf[49]["position_y"]], dtype=torch.float)
    theta = torch.tensor(fdf[49]["heading"], dtype=torch.float)
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    out = torch.linalg.norm(cur_pos - origin, dim=1) > MAP_RADIUS
    ids = [a for i, a in enumerate(ids) if not out[i]]
    if focal in ids:
        ids.remove(focal)
    ids = [focal] + ids                 # focal is ALWAYS node 0
    N = len(ids)
    df = df[df["track_id"].isin(ids)]

    pos = torch.zeros(N, TOT, 2)
    vel = torch.zeros(N, TOT, 2)         # 2D velocity (rotated)
    head = torch.zeros(N, TOT)
    attr = torch.zeros(N, dtype=torch.long)
    pad = torch.ones(N, TOT, dtype=torch.bool)

    for aid, adf in df.groupby("track_id"):
        i = ids.index(aid)
        steps = [ts.index(t) for t in adf["timestep"]]
        otype = OBJECT_TYPE[adf["object_type"].values[0]]
        attr[i] = otype
        pad[i, steps] = False
        if pad[i, 49] or otype in IGNORE:
            pad[i, 50:] = True
        xy = torch.from_numpy(np.stack([adf["position_x"].values, adf["position_y"].values], -1)).float()
        v = torch.from_numpy(adf[["velocity_x", "velocity_y"]].values).float()
        h = torch.from_numpy(adf["heading"].values).float()
        pos[i, steps] = torch.matmul(xy - origin, rotate_mat)
        vel[i, steps] = torch.matmul(v, rotate_mat)          # rotate velocity as a vector
        head[i, steps] = (h - theta + np.pi) % (2 * np.pi) - np.pi

    # --- lanes: centerline -> vector (HiVT format) ---
    cls, inters = get_lanes(am, origin, rotate_mat)
    lane_vectors, lane_pos_flat, is_inter_seg = [], [], []
    for cl, it in zip(cls, inters):
        m = (cl.abs() < MAP_RADIUS).all(-1)                  # drop points outside the bbox
        cl = cl[m]
        if cl.shape[0] < 2:
            continue
        lane_vectors.append(cl[1:] - cl[:-1])
        lane_pos_flat.append(cl[:-1])
        is_inter_seg.append(torch.full((cl.shape[0] - 1,), it))
    lane_vectors = torch.cat(lane_vectors) if lane_vectors else torch.zeros(0, 2)
    lane_pos_flat = torch.cat(lane_pos_flat) if lane_pos_flat else torch.zeros(0, 2)
    is_intersections = torch.cat(is_inter_seg).to(torch.uint8) if is_inter_seg else torch.zeros(0, dtype=torch.uint8)
    L = lane_vectors.size(0)
    # AV2 has no turn_direction/traffic_control -> default 0 (NONE)
    turn_directions = torch.zeros(L, dtype=torch.uint8)
    traffic_controls = torch.zeros(L, dtype=torch.uint8)

    # --- keep history positions, convert x to displacement (HiVT convention) ---
    positions = pos[:, :HIST].clone()
    x_vel = vel[:, :HIST].clone()                            # input velocity (no diff needed)
    x = pos.clone()
    x[:, HIST:] = torch.where((pad[:, HIST - 1].unsqueeze(-1) | pad[:, HIST:]).unsqueeze(-1),
                              torch.zeros(N, FUT, 2), x[:, HIST:] - x[:, HIST - 1].unsqueeze(-2))
    x[:, 1:HIST] = torch.where((pad[:, :HIST - 1] | pad[:, 1:HIST]).unsqueeze(-1),
                               torch.zeros(N, HIST - 1, 2), x[:, 1:HIST] - x[:, :HIST - 1])
    x[:, 0] = 0
    y = None if mode == "test" else x[:, HIST:]

    # bos_mask + rotate_angles (using the real heading)
    bos = torch.zeros(N, HIST, dtype=torch.bool)
    bos[:, 0] = ~pad[:, 0]
    bos[:, 1:HIST] = pad[:, :HIST - 1] & ~pad[:, 1:HIST]
    rotate_angles = head[:, HIST - 1].clone()                # AV2 heading at the current step

    # --- graph: full agent-agent edges + lane-actor ---
    edge_index = torch.LongTensor(list(permutations(range(N), 2))).t().contiguous()
    if L > 0:
        la_index = torch.LongTensor(list(product(range(L), range(N)))).t().contiguous()
        la_vec = lane_pos_flat.repeat_interleave(N, 0) - positions[:, HIST - 1].repeat(L, 1)
        keep = torch.linalg.norm(la_vec, dim=-1) < LOCAL_RADIUS
        la_index, la_vec = la_index[:, keep], la_vec[keep]
    else:
        la_index, la_vec = torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2)

    return TemporalData(
        x=x[:, :HIST], positions=positions, edge_index=edge_index, y=y, num_nodes=N,
        padding_mask=pad, bos_mask=bos, rotate_angles=rotate_angles,
        lane_vectors=lane_vectors, is_intersections=is_intersections,
        turn_directions=turn_directions, traffic_controls=traffic_controls,
        lane_actor_index=la_index, lane_actor_vectors=la_vec,
        seq_id=sid, av_index=0, agent_index=0,
        x_vel=x_vel, x_attr=attr, origin=origin.unsqueeze(0), theta=theta,
    )


def _worker(task):
    """Top-level (picklable) worker for multiprocessing.Pool. task = (parquet, split, save_dir)."""
    f, split, save_dir = task
    sid = os.path.basename(os.path.dirname(f))          # folder name = scenario id
    out = os.path.join(save_dir, sid + ".pt")
    if os.path.exists(out):
        return None                                     # resume: skip scenes already done
    try:
        torch.save(process(Path(f), split), out)
        return None
    except Exception as e:
        return f"ERR {sid}: {repr(e)[:100]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder containing <split>/<scenario>/*.parquet")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out_subdir", default="hivt-av2")
    ap.add_argument("-p", "--parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only process N scenes (quick test)")
    args = ap.parse_args()

    root = Path(args.data_root)
    split_dir = root / args.split
    # list quickly with scandir (no recursive rglob): each scenario is one subfolder
    ids = [e.name for e in os.scandir(split_dir) if e.is_dir()]
    files = [str(split_dir / i / f"scenario_{i}.parquet") for i in sorted(ids)]
    if args.limit:
        files = files[: args.limit]
    save_dir = root / args.out_subdir / args.split
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} scenario -> {save_dir}", flush=True)

    tasks = [(f, args.split, str(save_dir)) for f in files]
    from tqdm import tqdm
    if args.parallel:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_worker, tasks, chunksize=20), total=len(tasks)):
                if r:
                    print(r, flush=True)
    else:
        for t in tqdm(tasks):
            r = _worker(t)
            if r:
                print(r, flush=True)


if __name__ == "__main__":
    main()
