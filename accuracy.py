import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------- Hardcoded File Paths ----------------
file1 = r"D:\prakhar\sprint00\simul1\xyz_files\frame_001100.xyz"
file2 = r"D:\prakhar\sprint00\simul1\xyz_files\frame_001000.xyz"
#"D:\prakhar\sprint01\inference_results\frame0000_predicted.xyz
# ---------------- Function to Load Coordinates ----------------
def load_xyz_coords(filepath):
    """
    Loads an XYZ file that has only coordinates (no atom labels).
    Skips first two header lines.
    Returns Nx3 numpy array.
    """
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]  # Skip header lines
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = map(float, parts[:3])
                    coords.append([x, y, z])
                except ValueError:
                    continue  # skip invalid lines
    return np.array(coords)

# ---------------- Main Comparison Function ----------------
def compare_xyz_coords(coords1, coords2, threshold=0.1):
    """
    Matches each point in coords1 to its nearest neighbor in coords2.
    Returns DataFrame of differences and similarity percentage.
    """
    tree = cKDTree(coords2)
    distances, indices = tree.query(coords1)

    diffs = coords2[indices] - coords1
    dx, dy, dz = diffs[:, 0], diffs[:, 1], diffs[:, 2]

    df = pd.DataFrame({
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'distance': distances
    })

    # Similarity calculation
    close_matches = np.sum(distances < threshold)
    similarity = (close_matches / len(coords1)) * 100

    return df, similarity

# ---------------- Run ----------------
coords1 = load_xyz_coords(file1)
coords2 = load_xyz_coords(file2)

if coords1.shape[0] == 0 or coords2.shape[0] == 0:
    print("Error: One of the XYZ files has no valid coordinates.")
    exit()

diff_df, similarity_percentage = compare_xyz_coords(coords1, coords2, threshold=0.1)

print(diff_df.describe())
print(f"\nSimilarity: {similarity_percentage:.2f}% of atoms are within 0.1 Å")
