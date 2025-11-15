import argparse
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align a SLAM point cloud and trajectory with GPS ground truth from a CSV file."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file that contains GPS (state_*) and SLAM (slam_*) columns.")
    parser.add_argument(
        "--map",
        dest="map_path",
        type=Path,
        default=None,
        help="Optional SLAM point-cloud map (PCD) to transform to global coordinates.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for generated artifacts (default: derived from CSV filename).",
    )
    parser.add_argument(
        "--calib",
        type=float,
        default=None,
        help="Calibration distance in meters. Only the first N meters of the trajectory are used to estimate the similarity transform; the rest is treated as test data.",
    )
    return parser.parse_args()


def load_point_cloud(map_path: Path):
    if not map_path.exists():
        raise FileNotFoundError(f"Point-cloud map not found: {map_path}")
    pcd = o3d.io.read_point_cloud(str(map_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("Point cloud is empty.")
    print("🔍 PCD File Info")
    print(f"- Number of points: {points.shape[0]}")
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    print(f"- X range: {min_bounds[0]:.2f} to {max_bounds[0]:.2f} meters")
    print(f"- Y range: {min_bounds[1]:.2f} to {max_bounds[1]:.2f} meters")
    print(f"- Z range: {min_bounds[2]:.2f} to {max_bounds[2]:.2f} meters")
    return points


def prepare_transformers():
    return (
        Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True),
        Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True),
    )


def extract_trajectories(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = [
        "state_lon",
        "state_lat",
        "state_alt",
        "slam_x",
        "slam_y",
        "slam_z",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df_numeric = df.copy()
    for col in required_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    valid = df_numeric.dropna(subset=required_cols)
    if len(valid) < 3:
        raise ValueError("Need at least three rows with both GPS and SLAM data to compute a rotation.")

    gps_lonlatalt = valid[["state_lon", "state_lat", "state_alt"]].to_numpy()
    slam_xyz = valid[["slam_x", "slam_y", "slam_z"]].to_numpy()
    timestamps = valid.get("slam_timestamp_s")

    return gps_lonlatalt, slam_xyz, timestamps


def compute_calibration_length(gps_lonlatalt, calib_distance_m):
    n = len(gps_lonlatalt)
    if n == 0:
        return 0, 0.0

    lon0, lat0 = gps_lonlatalt[0, 0], gps_lonlatalt[0, 1]
    local_proj = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("epsg:4326", local_proj, always_xy=True)
    xy = np.array([transformer.transform(lon, lat) for lon, lat in gps_lonlatalt[:, :2]])

    diffs = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(diffs)))

    if calib_distance_m is None or calib_distance_m <= 0:
        return n, cumulative[-1] if len(cumulative) > 0 else 0.0

    idx = np.searchsorted(cumulative, calib_distance_m, side="right")
    idx = max(3, idx)
    calib_len = min(n, idx)
    actual_distance = cumulative[calib_len - 1]
    return calib_len, actual_distance


def align_frames(gps_ecef, slam_xyz):
    first_slam = slam_xyz[0]
    first_gps = gps_ecef[0]

    slam_centered = slam_xyz - first_slam
    gps_centered = gps_ecef - first_gps

    if np.linalg.norm(slam_centered, axis=1).max() == 0:
        raise ValueError("SLAM trajectory has no spatial extent after centering.")

    rotation, _ = R.align_vectors(gps_centered, slam_centered)
    slam_rotated = rotation.apply(slam_centered)

    denom = np.sum(slam_rotated * slam_rotated)
    if denom == 0:
        raise ValueError("Cannot compute scale: SLAM data collapsed to a point.")
    scale = np.sum(slam_rotated * gps_centered) / denom

    translation = first_gps

    print("Rotation Matrix (from SLAM to GPS):")
    print(rotation.as_matrix())
    print(f"Scale Factor: {scale:.6f}")
    print("Translation Vector:", translation)

    return rotation, scale, translation, first_slam


def transform_points_to_ecef(slam_xyz, rotation, scale, translation, first_slam):
    slam_centered = slam_xyz - first_slam
    slam_rotated = rotation.apply(slam_centered)
    return scale * slam_rotated + translation


def transform_map(points, rotation, scale, translation, first_slam, ecef_to_wgs84):
    slam_map_ecef = transform_points_to_ecef(points, rotation, scale, translation, first_slam)
    return np.array([ecef_to_wgs84.transform(*pt) for pt in slam_map_ecef])


def ecef_to_wgs84_batch(points_ecef, transformer):
    return np.array([transformer.transform(*pt) for pt in points_ecef])


def save_artifacts(prefix, df_local, df_global, df_traj):
    if df_local is not None:
        local_path = f"{prefix}_slam_map_local_coords.csv"
        df_local.to_csv(local_path, index=False)
        print(f"✅ Saved local coordinates to {local_path}")

    if df_global is not None:
        global_path = f"{prefix}_slam_map_global_coords.csv"
        df_global.to_csv(global_path, index=False)
        print(f"✅ Saved global map coordinates to {global_path}")

    if df_traj is not None:
        traj_path = f"{prefix}_trajectory_global_coords.csv"
        df_traj.to_csv(traj_path, index=False)
        print(f"✅ Saved transformed trajectory to {traj_path}")


def build_map(prefix, df_traj, gps_lonlatalt, df_global=None):
    if df_global is not None and not df_global.empty:
        start_lat = df_global["lat"].iloc[0]
        start_lon = df_global["lon"].iloc[0]
    else:
        start_lat = df_traj["lat"].iloc[0]
        start_lon = df_traj["lon"].iloc[0]

    m = folium.Map(location=[start_lat, start_lon], zoom_start=18, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    if df_global is not None:
        for lat, lon in zip(df_global["lat"][::100], df_global["lon"][::100]):
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
            ).add_to(m)

    for lon, lat, _ in gps_lonlatalt:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.6,
        ).add_to(m)

    trajectory = list(zip(df_traj["lat"], df_traj["lon"]))
    folium.PolyLine(trajectory, color="yellow", weight=1.8).add_to(m)
    stride = max(1, len(trajectory) // 500)
    for lat, lon in trajectory[::stride]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="yellow",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.6,
        ).add_to(m)
    html_path = f"{prefix}_trajectory_map.html"
    m.save(html_path)
    print(f"✅ Saved interactive map to {html_path}")
    return html_path


def plot_static(gps_calib, gps_test, traj_test):
    plt.figure(figsize=(10, 10))

    if len(gps_calib) > 1:
        plt.plot(
            gps_calib[:, 0],
            gps_calib[:, 1],
            linestyle="--",
            color="gray",
            linewidth=2,
            label="GPS Calibration",
        )

    if len(gps_test) > 1:
        plt.plot(
            gps_test[:, 0],
            gps_test[:, 1],
            linestyle="-",
            color="blue",
            linewidth=2,
            label="GPS Test",
        )

    if len(traj_test) > 1:
        plt.plot(
            traj_test[:, 0],
            traj_test[:, 1],
            linestyle="--",
            color="red",
            linewidth=2,
            label="SLAM→GPS (Test)",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Calibration/Test Split")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    csv_path = args.csv_path
    prefix = args.output_prefix or csv_path.stem

    # Rotate SLAM axes to ENU-like frame.
    T_fix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    map_points_fixed = None
    df_local = None

    if args.map_path is not None:
        points = load_point_cloud(args.map_path)
        map_points_fixed = points @ T_fix.T
        df_local = pd.DataFrame(map_points_fixed, columns=["x", "y", "z"])

    wgs84_to_ecef, ecef_to_wgs84 = prepare_transformers()
    gps_lonlatalt, slam_xyz, _ = extract_trajectories(csv_path)

    slam_xyz_fixed = slam_xyz @ T_fix.T
    gps_ecef = np.array([wgs84_to_ecef.transform(lon, lat, alt) for lon, lat, alt in gps_lonlatalt])

    calib_len, calib_distance = compute_calibration_length(gps_lonlatalt, args.calib)
    print(
        f"Calibration samples: {calib_len} (≈ {calib_distance:.2f} m). Test samples: {len(gps_ecef) - calib_len}."
    )

    gps_ecef_calib = gps_ecef[:calib_len]
    slam_calib = slam_xyz_fixed[:calib_len]

    rotation, scale, translation, first_slam = align_frames(gps_ecef_calib, slam_calib)

    trajectory_ecef = transform_points_to_ecef(
        slam_xyz_fixed, rotation, scale, translation, first_slam
    )
    trajectory_global = ecef_to_wgs84_batch(trajectory_ecef, ecef_to_wgs84)

    df_global = None
    if map_points_fixed is not None:
        slam_map_global = transform_map(
            map_points_fixed, rotation, scale, translation, first_slam, ecef_to_wgs84
        )
        df_global = pd.DataFrame(slam_map_global, columns=["lon", "lat", "alt"])
    df_traj = pd.DataFrame(trajectory_global, columns=["lon", "lat", "alt"])

    save_artifacts(prefix, df_local, df_global, df_traj)
    build_map(prefix, df_traj, gps_lonlatalt, df_global)

    gps_calib = gps_lonlatalt[:calib_len]
    gps_test = gps_lonlatalt[calib_len:]
    traj_test = trajectory_global[calib_len:]

    gps_calib_xy = gps_calib[:, :2] if len(gps_calib) else np.empty((0, 2))
    gps_test_xy = gps_test[:, :2] if len(gps_test) else np.empty((0, 2))
    traj_test_xy = traj_test[:, :2] if len(traj_test) else np.empty((0, 2))

    plot_static(gps_calib_xy, gps_test_xy, traj_test_xy)


if __name__ == "__main__":
    main()
