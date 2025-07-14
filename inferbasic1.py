import numpy as np
import tensorflow as tf
import pickle
import os

def load_model_and_scaler():
    """Load the trained model, scaler, and atom IDs"""
    # Load model without compilation, then compile manually
    model = tf.keras.models.load_model('molecular_prediction_model.h5', compile=False)
    model.compile(optimizer='adam', loss=chamfer_loss)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('atom_ids.pkl', 'rb') as f:
        atom_ids = pickle.load(f)
    
    return model, scaler, atom_ids

def chamfer_loss(y_true, y_pred):
    """Chamfer loss function (needed for model loading)"""
    coords_true = tf.reshape(y_true, (-1, tf.shape(y_true)[1] // 3, 3))
    coords_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1] // 3, 3))
    
    diff = tf.expand_dims(coords_pred, 2) - tf.expand_dims(coords_true, 1)
    distances = tf.reduce_sum(tf.square(diff), axis=3)
    
    min_dist_pred_to_true = tf.reduce_min(distances, axis=2)
    forward_loss = tf.reduce_mean(min_dist_pred_to_true, axis=1)
    
    min_dist_true_to_pred = tf.reduce_min(distances, axis=1)
    backward_loss = tf.reduce_mean(min_dist_true_to_pred, axis=1)
    
    chamfer_dist = forward_loss + backward_loss
    return tf.reduce_mean(chamfer_dist)

def parse_single_xyz(file_path, atom_ids):
    """Parse a single XYZ file and return coordinates for specified atom IDs"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        atom_dict = {}
        for line in lines[2:]:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                atom_id = int(parts[3])
                coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                atom_dict[atom_id] = coords
            except ValueError:
                continue
    
    frame = []
    for atom_id in atom_ids:
        if atom_id in atom_dict:
            frame.extend(atom_dict[atom_id])
        else:
            raise ValueError(f"Atom ID {atom_id} missing in file {file_path}")
    
    return np.array(frame)

def predict_future_frame(input_xyz_path, output_xyz_path):
    """Predict future molecular configuration from input XYZ file"""
    # Load model and preprocessing objects
    model, scaler, atom_ids = load_model_and_scaler()
    
    # Parse input XYZ file
    input_frame = parse_single_xyz(input_xyz_path, atom_ids)
    
    # Preprocess input
    input_scaled = scaler.transform(input_frame.reshape(1, -1))
    
    # Make prediction
    prediction_scaled = model.predict(input_scaled)
    
    # Inverse transform to get actual coordinates
    prediction = scaler.inverse_transform(prediction_scaled)
    
    # Write output XYZ file
    write_xyz_file(output_xyz_path, prediction[0], atom_ids)
    
    return prediction[0]

def write_xyz_file(file_path, coords, atom_ids):
    """Write coordinates to XYZ file"""
    num_atoms = len(atom_ids)
    
    with open(file_path, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("Predicted molecular configuration\n")
        
        for i, atom_id in enumerate(atom_ids):
            x, y, z = coords[i*3:(i+1)*3]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {atom_id}\n")

if __name__ == "__main__":
    input_file = "xyz_files/frame_1000.xyz"  # Replace with your input XYZ file
    output_file = "predicted_basic.xyz"  # Output file name
    
    try:
        predicted_coords = predict_future_frame(input_file, output_file)
        print(f"Prediction completed. Output saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")