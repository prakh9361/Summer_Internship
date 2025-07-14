import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from nn1 import MemristorCycleGNN # IMPORTANT: This now imports the corrected model
from torch_cluster import radius_graph
from tqdm import tqdm
from collections import OrderedDict

def load_model_robustly(model, model_path, device, strict=False):
    """
    Robustly load a model state dict, handling different checkpoint formats
    and architecture mismatches.
    
    Args:
        model: The model instance to load weights into
        model_path: Path to the saved model
        device: Device to load the model on
        strict: If True, requires exact key matching. If False, loads compatible weights only.
    
    Returns:
        tuple: (success: bool, message: str, loaded_keys: list, missing_keys: list)
    """
    try:
        print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict from different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Found enhanced checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"Previous best loss: {checkpoint.get('best_loss', 'unknown')}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Found checkpoint with 'state_dict' key")
            else:
                # Check if this is a model state dict or full checkpoint
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(checkpoint.keys())
                
                # If checkpoint has metadata keys, it's not a pure state dict
                metadata_keys = {'epoch', 'best_loss', 'model_config', 'training_config', 'optimizer_state_dict', 'scheduler_state_dict'}
                if metadata_keys.intersection(checkpoint_keys):
                    print("WARNING: Checkpoint contains metadata but no 'model_state_dict' key")
                    print("This checkpoint may be incompatible. Available keys:", list(checkpoint_keys)[:10])
                    return False, "Incompatible checkpoint format", [], []
                else:
                    # Assume it's a direct state dict
                    state_dict = checkpoint
                    print("Treating checkpoint as direct state dict")
        else:
            # Direct state dict
            state_dict = checkpoint
            print("Loading direct state dict")
        
        # Get current model's state dict for comparison
        model_state_dict = model.state_dict()
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Find compatible keys
        compatible_keys = model_keys.intersection(checkpoint_keys)
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        print(f"Model keys: {len(model_keys)}")
        print(f"Checkpoint keys: {len(checkpoint_keys)}")
        print(f"Compatible keys: {len(compatible_keys)}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        if len(compatible_keys) == 0:
            return False, "No compatible keys found between model and checkpoint", [], list(missing_keys)
        
        if len(compatible_keys) < len(model_keys) * 0.5:  # Less than 50% compatibility
            print("WARNING: Low compatibility between model and checkpoint")
            print("Missing keys (first 10):", list(missing_keys)[:10])
            print("Unexpected keys (first 10):", list(unexpected_keys)[:10])
            
            if strict:
                return False, "Insufficient key compatibility in strict mode", list(compatible_keys), list(missing_keys)
        
        # Create a filtered state dict with only compatible keys
        filtered_state_dict = OrderedDict()
        for key in compatible_keys:
            try:
                # Check if tensor shapes match
                if model_state_dict[key].shape == state_dict[key].shape:
                    filtered_state_dict[key] = state_dict[key]
                else:
                    print(f"Shape mismatch for key '{key}': model {model_state_dict[key].shape} vs checkpoint {state_dict[key].shape}")
                    missing_keys.add(key)
                    compatible_keys.remove(key)
            except Exception as e:
                print(f"Error processing key '{key}': {e}")
                missing_keys.add(key)
                if key in compatible_keys:
                    compatible_keys.remove(key)
        
        # Load the filtered state dict
        if len(filtered_state_dict) > 0:
            model.load_state_dict(filtered_state_dict, strict=False)
            success_msg = f"Successfully loaded {len(filtered_state_dict)}/{len(model_keys)} parameters"
            print(success_msg)
            return True, success_msg, list(compatible_keys), list(missing_keys)
        else:
            return False, "No compatible parameters found", [], list(missing_keys)
            
    except Exception as e:
        error_msg = f"Error loading checkpoint: {str(e)}"
        print(error_msg)
        return False, error_msg, [], []

def load_xyz_file(filepath):
    """Loads an xyz file into a numpy array."""
    try:
        return pd.read_csv(filepath, sep='\s+', header=None, skiprows=2, usecols=[0,1,2], engine='python').values.astype(np.float32)
    except Exception:
        # Fallback for different formats
        return pd.read_csv(filepath, sep=',', header=None).values.astype(np.float32)

def create_graph(coords, max_distance=5.0, max_neighbors=32):
    """Creates a graph representation from atomic coordinates."""
    coords_tensor = torch.from_numpy(coords).float()
    
    # Use radius_graph to find neighbors within a certain distance
    edge_index = radius_graph(coords_tensor, r=max_distance, loop=False, max_num_neighbors=max_neighbors)
    
    row, col = edge_index
    edge_vectors = coords_tensor[row] - coords_tensor[col]
    edge_attr = torch.norm(edge_vectors, p=2, dim=1).unsqueeze(1)

    return coords_tensor, edge_index, edge_attr

def save_xyz(coords, filename):
    """Saves atomic coordinates to an XYZ file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write("Predicted atomic positions by GNN\n")
        # Assuming atom type '1', adjust if necessary
        for coord in coords:
            f.write(f"1 {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n") 

def run_inference(
    model_path,
    input_xyz_path,
    input_metadata,
    target_metadata,
    output_xyz_path,
    hidden_dim=32,
    num_layers=5,
    dropout=0.2,
    strict_loading=False
):
    """Runs inference using the trained GNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load input structure and create graph
    print(f"Loading input file: {input_xyz_path}")
    initial_coords = load_xyz_file(input_xyz_path)
    coords_tensor, edge_index, edge_attr = create_graph(initial_coords)
    num_atoms = initial_coords.shape[0]

    # Prepare metadata tensors by repeating them for each atom
    current_conditions = torch.tensor([[input_metadata['volt'], input_metadata['temp'], input_metadata['time']]] * num_atoms, dtype=torch.float32)
    target_conditions = torch.tensor([[target_metadata['volt'], target_metadata['temp'], target_metadata['time']]] * num_atoms, dtype=torch.float32)

    # Assemble the data into a PyTorch Geometric Data object
    data = Data(
        x=coords_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        current_conditions=current_conditions,
        target_conditions=target_conditions
    ).to(device)

    # Load the trained model
    print("Initializing model...")
    model = MemristorCycleGNN(
        input_dim=3,
        edge_dim=1,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Use robust loading
    print("Loading model weights...")
    success, message, loaded_keys, missing_keys = load_model_robustly(
        model, model_path, device, strict=strict_loading
    )
    
    if not success:
        print(f"Failed to load model: {message}")
        print("Trying with a fresh model (no pre-trained weights)...")
        print("WARNING: Using untrained model - results may not be meaningful!")
    else:
        print(f"Model loading result: {message}")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} (these layers will use random initialization)")
    
    model.eval()  # Set model to evaluation mode

    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        try:
            predicted_coords = model(
                data.x, data.edge_index, data.edge_attr,
                data.current_conditions, data.target_conditions
            )
            predicted_coords_numpy = predicted_coords.cpu().numpy()
        except Exception as e:
            print(f"Error during inference: {e}")
            print("This might be due to architecture mismatch between saved model and current model definition.")
            return

    # Compare ranges to see if there was a change
    initial_x_range = np.max(initial_coords[:,0]) - np.min(initial_coords[:,0])
    predicted_x_range = np.max(predicted_coords_numpy[:,0]) - np.min(predicted_coords_numpy[:,0])
    print(f"Initial X-range: {initial_x_range:.4f}")
    print(f"Predicted X-range: {predicted_x_range:.4f}")
    
    # Check the displacement
    displacement = np.mean(np.linalg.norm(predicted_coords_numpy - initial_coords, axis=1))
    print(f"Mean atomic displacement: {displacement:.6f}")

    # Save the prediction
    save_xyz(predicted_coords_numpy, output_xyz_path)
    print(f"Predicted structure saved to {output_xyz_path}")

def diagnose_checkpoint(model_path):
    """
    Utility function to diagnose what's in a checkpoint file
    """
    print(f"Diagnosing checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        if 'model_config' in checkpoint:
            print("Model config:", checkpoint['model_config'])
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Model state dict has {len(state_dict)} keys")
            print("First 10 model keys:", list(state_dict.keys())[:10])
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"State dict has {len(state_dict)} keys")
            print("First 10 keys:", list(state_dict.keys())[:10])
        else:
            print(f"Checkpoint dict has {len(checkpoint)} keys")
            print("First 10 keys:", list(checkpoint.keys())[:10])
    else:
        print("Checkpoint is not a dictionary - might be direct state dict")
        if hasattr(checkpoint, 'keys'):
            print(f"Has {len(checkpoint)} keys")

if __name__ == "__main__":
    # Optional: Diagnose the checkpoint first
    model_path = r"D:\prakhar\sprint01\best_model.pt"
    print("=== Checkpoint Diagnosis ===")
    diagnose_checkpoint(model_path)
    print("\n=== Running Inference ===")
    
    run_inference(
        model_path=model_path,
        input_xyz_path=r"D:\prakhar\sprint01\simul1\xyz_files\frame_001000.xyz",
        input_metadata={'volt': 36.5, 'temp': 1030.35, 'time': 500.0},
        target_metadata={'volt': 37.23, 'temp': 1053.13, 'time': 550.0},
        output_xyz_path=r"D:\prakhar\sprint01\inference_results\frame_001100_predicted.xyz",
        strict_loading=False  # Set to True if you want strict key matching
    )