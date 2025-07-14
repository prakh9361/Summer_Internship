import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import glob
import pickle
from scipy.spatial.distance import cdist


def parse_xyz_files(directories):
    xyz_files = []
    for directory in directories:
        xyz_files.extend(sorted(glob.glob(os.path.join(directory, "*.xyz"))))
    
    if len(xyz_files) < 101:
        raise ValueError("Not enough XYZ files to create input-output pairs (need at least 101 frames).")

    # Read the first file to get the list of atom IDs
    with open(xyz_files[0], 'r') as f:
        lines = f.readlines()
        atom_data = [line.split() for line in lines[2:] if len(line.split()) == 4]
        atom_ids = set([int(line[3]) for line in atom_data])

    # Find common atom IDs across all files
    for file in xyz_files[1:]:
        with open(file, 'r') as f:
            lines = f.readlines()
            current_ids = set([int(line.split()[3]) for line in lines[2:] if len(line.split()) == 4])
            atom_ids = atom_ids.intersection(current_ids)

    atom_ids = sorted(list(atom_ids))
    print(f"Found {len(atom_ids)} common atom IDs: {atom_ids}")

    # Parse coordinates for common atom IDs
    frames = []
    for file in xyz_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            atom_dict = {}
            for line in lines[2:]:
                parts = line.split()
                if len(parts) != 4:
                    print(f"Skipping malformed line in {file}: {line.strip()}")
                    continue
                try:
                    atom_id = int(parts[3])
                    coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                    atom_dict[atom_id] = coords
                except ValueError as e:
                    print(f"Error parsing line in {file}: {line.strip()} | Error: {e}")
                    continue
            # Create frame data in order of atom_ids
            frame = []
            for atom_id in atom_ids:
                if atom_id in atom_dict:
                    frame.extend(atom_dict[atom_id])
                else:
                    raise ValueError(f"Atom ID {atom_id} missing in file {file}")
            frames.append(frame)

    return np.array(frames), atom_ids




"""def multi_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Collapse penalty for atoms closer than 2.9 angstrom
    coords_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1] // 3, 3))
    
    # Calculate pairwise distances
    expanded_a = tf.expand_dims(coords_pred, 2)
    expanded_b = tf.expand_dims(coords_pred, 1)
    distances = tf.sqrt(tf.reduce_sum(tf.square(expanded_a - expanded_b), axis=3))
    
    # Mask to exclude diagonal (self-distances)
    mask = 1.0 - tf.eye(tf.shape(coords_pred)[1])
    distances = distances * mask
    
    # Penalty for distances below 2.9 angstrom
    collapse_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, 2.9 - distances)))
    
    return mse_loss + collapse_penalty
"""
def combined_loss(y_true, y_pred):
    # Reshape to (batch_size, num_atoms, 3)
    coords_true = tf.reshape(y_true, (-1, tf.shape(y_true)[1] // 3, 3))
    coords_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1] // 3, 3))
    
    # Chamfer distance
    diff = tf.expand_dims(coords_pred, 2) - tf.expand_dims(coords_true, 1)
    distances = tf.reduce_sum(tf.square(diff), axis=3)
    
    min_dist_pred_to_true = tf.reduce_min(distances, axis=2)
    forward_loss = tf.reduce_mean(min_dist_pred_to_true, axis=1)
    
    min_dist_true_to_pred = tf.reduce_min(distances, axis=1)
    backward_loss = tf.reduce_mean(min_dist_true_to_pred, axis=1)
    
    chamfer_dist = tf.reduce_mean(forward_loss + backward_loss)
    
    mse_loss = tf.reduce_mean(tf.square(coords_pred - coords_true))

    # Collapse penalty
    expanded_a = tf.expand_dims(coords_pred, 2)
    expanded_b = tf.expand_dims(coords_pred, 1)
    pairwise_distances = tf.sqrt(tf.reduce_sum(tf.square(expanded_a - expanded_b), axis=3))
    
    mask = 1.0 - tf.eye(tf.shape(coords_pred)[1])
    pairwise_distances = pairwise_distances * mask
    
    collapse_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, 2.9 - pairwise_distances)))
    
    # Volume constraint (using convex hull approximation)
    def calculate_volume(coords):
        # Simple bounding box volume approximation
        min_coords = tf.reduce_min(coords, axis=1)  # (batch_size, 3)
        max_coords = tf.reduce_max(coords, axis=1)  # (batch_size, 3)
        volume = tf.reduce_prod(max_coords - min_coords, axis=1)  # (batch_size,)
        return volume
    
    volume_true = calculate_volume(coords_true)
    volume_pred = calculate_volume(coords_pred)
    
    volume_ratio = volume_pred / (volume_true + 1e-8)  # Avoid division by zero
    volume_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, tf.abs(volume_ratio - 1.0) - 0.04)))
    
    return chamfer_dist + collapse_penalty + volume_penalty + mse_loss

# Function to create input-output pairs (frame t -> frame t+100)
def create_dataset(frames, look_ahead=100):
    X, y = [], []
    for i in range(len(frames) - look_ahead):
        X.append(frames[i])
        y.append(frames[i + look_ahead])
    return np.array(X), np.array(y)

# Function to build the ANN model
def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim)  # Output same size as input (3N coordinates)
    ])
    model.compile(optimizer='adam', loss=combined_loss)
    return model

# Main processing
def main():
    # Multiple directories containing XYZ files
    xyz_dirs = ["xyz_files"]  # Add your directories here
    
    # Parse XYZ files
    frames, atom_ids = parse_xyz_files(xyz_dirs)
    num_atoms = len(atom_ids)
    print(f"Parsed {len(frames)} frames with {num_atoms} atoms each.")

    # Normalize the data
    scaler = StandardScaler()
    frames_scaled = scaler.fit_transform(frames)

    # Create input-output pairs
    X, y = create_dataset(frames_scaled)
    print(f"Created {len(X)} input-output pairs.")

    # Split into training and validation sets
    train_size = int(1 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # Build and train the model
    model = build_model(X.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

    # Save the model and scaler
    model.save('molecular_prediction_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('atom_ids.pkl', 'wb') as f:
        pickle.dump(atom_ids, f)
    
    print("Model saved as 'molecular_prediction_model.h5'")
    print("Scaler saved as 'scaler.pkl'")
    print("Atom IDs saved as 'atom_ids.pkl'")

if __name__ == "__main__":
    main()