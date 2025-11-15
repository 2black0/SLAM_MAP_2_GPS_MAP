### README: SLAM to GPS Coordinate Transformation

---

#### **Overview**
This project aligns a SLAM (Simultaneous Localization and Mapping) trajectory and point cloud data with a GPS trajectory to transform SLAM points into a global coordinate system (WGS84). The process involves computing a rotation matrix and translation vector to align the SLAM data with the GPS data, enabling visualization and analysis in a global reference frame.

---

#### **Features**
- Converts SLAM map points from a local coordinate system to a global WGS84 coordinate system.
- Aligns SLAM trajectory with GPS trajectory using:
  - Full trajectory alignment (via `R.align_vectors`).
  - Two-point vector alignment (optional).
- Saves transformed SLAM map and trajectory as CSV files.
- Visualizes the results on an interactive map (using Folium) and a 2D plot (using Matplotlib).

---

#### **Dependencies**
The following Python libraries are required:
- `open3d`: For reading and processing point cloud data.
- `numpy`: For numerical operations.
- `pandas`: For handling tabular data.
- `pyproj`: For coordinate transformations (WGS84 ↔ ECEF).
- `matplotlib`: For 2D plotting.
- `folium`: For creating interactive maps.
- `scipy`: For computing rotations and alignments.

Install dependencies using:
```bash
pip install open3d numpy pandas pyproj matplotlib folium scipy
```

---

#### **Input Files**
1. **SLAM Map**:
   - A `.pcd` file containing the SLAM map points (e.g., `map.pcd`).
2. **SLAM Trajectory**:
   - A text file containing SLAM poses in KITTI format (e.g., `output_120_d_poses_kitti.txt`).
3. **GPS Trajectory**:
   - A CSV file containing GPS data with `Latitude` and `Longitude` columns (e.g., `gnss_data_20250324_103623.csv`).

---

#### **Output Files**
1. **SLAM Map (Global Coordinates)**:
   - A CSV file (`slam_map_global_coords_corrected.csv`) containing the SLAM map points transformed to WGS84 coordinates.
2. **SLAM Trajectory (Global Coordinates)**:
   - A CSV file (`trajectory_global_coords_transformed.csv`) containing the SLAM trajectory transformed to WGS84 coordinates.
3. **Interactive Map**:
   - An HTML file (`trajectory_with_slam_and_interpolated_gps_subset.html`) visualizing the SLAM map, GPS trajectory, and SLAM trajectory on a satellite map.

---

#### **How It Works**
1. **Load SLAM Map and Trajectory**:
   - Reads the SLAM map (`map.pcd`) and trajectory (`output_120_d_poses_kitti.txt`).
   - Applies a fixed rotation (`T_fix`) to align the SLAM coordinate system with the ENU (East-North-Up) frame.

2. **Load and Interpolate GPS Trajectory**:
   - Reads the GPS trajectory (`gnss_data_20250324_103623.csv`) and converts it to ECEF coordinates.
   - Interpolates the GPS trajectory to match the length of the SLAM trajectory.

3. **Compute Transformation**:
   - **Translation**:
     - Computes the translation vector using the first SLAM and GPS points.
   - **Rotation**:
     - Computes the rotation matrix using either:
       - Full trajectory alignment (`R.align_vectors`).
       - Two-point vector alignment (optional).

4. **Transform SLAM Data**:
   - Applies the computed rotation and translation to the SLAM map and trajectory.
   - Converts the transformed data to WGS84 coordinates.

5. **Save and Visualize**:
   - Saves the transformed SLAM map and trajectory as CSV files.
   - Visualizes the results on an interactive map (Folium) and a 2D plot (Matplotlib).

---

#### **Usage**
1. **Run the Script**:
   - Execute the Python script to process the input files and generate the outputs:
     ```bash
     python Map_coordinates.py
     ```

2. **View Outputs**:
   - Open the generated CSV files to inspect the transformed SLAM map and trajectory.
   - Open the HTML file (`trajectory_with_slam_and_interpolated_gps_subset.html`) in a browser to view the interactive map.


