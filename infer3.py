import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from torch_geometric.nn import NNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

# Try to import GPU-accelerated graph functions
try:
    from torch_cluster import radius_graph, knn_graph
    TORCH_CLUSTER_AVAILABLE = True
    print("torch_cluster is available. Using GPU for graph creation if available.")
except ImportError:
    TORCH_CLUSTER_AVAILABLE = False
    print("torch_cluster not available. Using CPU fallback for graph creation.")

# =====================================================================================
# == PASTE REQUIRED CLASSES FROM YOUR TRAINING SCRIPT ==
#
# To make this script standalone, we need the definitions for the model
# and the data processor.
# =====================================================================================

# ----------------- Enhanced GNN Model for Memristor -----------------
class MemristorGNN(nn.Module):
    """
    Enhanced GNN model specifically designed for memristor filament formation.
    Includes voltage-aware processing and constraint-aware predictions.
    (Copied from the training script for consistency)
    """
    def __init__(self, input_dim=10, edge_dim=1, hidden_dim=32, output_dim=3,
                 num_layers=4, dropout=0.1, voltage_embedding_dim=8):
        super(MemristorGNN, self).__init__()
        self.dropout = dropout
        self.voltage_embedding_dim = voltage_embedding_dim

        # Voltage embedding network
        self.voltage_encoder = nn.Sequential(
            nn.Linear(1, voltage_embedding_dim),
            nn.ReLU(),
            nn.Linear(voltage_embedding_dim, voltage_embedding_dim)
        )

        # Enhanced input processing with voltage context
        enhanced_input_dim = input_dim + voltage_embedding_dim
        
        self.nn_input = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * enhanced_input_dim)
        )
        self.input_conv = NNConv(enhanced_input_dim, hidden_dim, self.nn_input, aggr='mean')
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Hidden layers with residual connections
        self.hidden_convs = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()

        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.hidden_convs.append(NNConv(hidden_dim, hidden_dim, nn_hidden, aggr='mean'))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer with constraint awareness
        self.nn_output = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * output_dim)
        )
        self.output_conv = NNConv(hidden_dim, output_dim, self.nn_output, aggr='mean')
        
        # Displacement scaling network
        self.displacement_scaler = nn.Sequential(
            nn.Linear(hidden_dim + voltage_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Scale displacements between 0 and 1
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        # Store original coordinates
        original_coords = x[:, :3]
        
        # Extract voltage information (assuming it's the 3rd from last feature)
        voltage = x[:, -3].unsqueeze(1)
        
        # Create voltage embeddings
        voltage_emb = self.voltage_encoder(voltage)
        
        # Enhance input features with voltage context
        enhanced_x = torch.cat([x, voltage_emb], dim=1)

        # Input layer
        x_processed = self.input_conv(enhanced_x, edge_index, edge_attr)
        x_processed = self.input_bn(x_processed)
        x_processed = F.relu(x_processed)
        x_processed = F.dropout(x_processed, p=self.dropout, training=self.training)

        # Hidden layers with residual connections
        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            residual = x_processed
            x_processed = conv(x_processed, edge_index, edge_attr)
            x_processed = bn(x_processed)
            x_processed = F.relu(x_processed)
            x_processed = F.dropout(x_processed, p=self.dropout, training=self.training)
            x_processed = x_processed + residual

        # Output layer produces raw displacement
        raw_displacement = self.output_conv(x_processed, edge_index, edge_attr)
        
        # Scale displacements based on voltage magnitude and local environment
        context = torch.cat([x_processed, voltage_emb], dim=1)
        displacement_scale = self.displacement_scaler(context) * 0.5  # Max 0.5 Angstrom displacement
        
        # Apply scaled displacement
        scaled_displacement = raw_displacement * displacement_scale
        
        # Final prediction: add scaled displacement to original coordinates
        output_coords = original_coords + scaled_displacement
        
        return output_coords

# ----------------- Data Processor for Inference -----------------
class EnhancedAtomicDataProcessor:
    """
    Handles the pre-processing of a single raw LAMMPS data file into a PyG Data object.
    (Copied from the training script for consistency)
    """
    def __init__(self, k_neighbors=8, max_distance=3.5):
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

    def _load_xyz_file(self, filepath):
        """Loads a single XYZ file, skipping header lines."""
        with open(filepath, 'r') as f:
            lines = f.readlines()[2:]
            data = [line.strip().split() for line in lines]
            coords = [[float(row[-3]), float(row[-2]), float(row[-1])] for row in data]
        return np.array(coords, dtype=np.float32)

    def _create_graph_from_coords(self, coords, device):
        """Creates graph edges and attributes from coordinates."""
        if not torch.is_tensor(coords):
            coords = torch.from_numpy(coords).float()
        coords = coords.to(device)
        edge_index = radius_graph(coords, r=self.max_distance, loop=False, max_num_neighbors=32)
        if edge_index.numel() == 0:
            print(f"Warning: No neighbors found with radius {self.max_distance}. Falling back to 2-NN graph.")
            edge_index = knn_graph(coords, k=min(2, len(coords)), loop=False)
        row, col = edge_index
        edge_vectors = coords[row] - coords[col]
        edge_attr = torch.norm(edge_vectors, p=2, dim=1).unsqueeze(1)
        return edge_index, edge_attr

    def _create_node_features(self, positions, voltage, temperature, time):
        """Creates the 10-dimensional node feature vector."""
        if not torch.is_tensor(positions):
            positions = torch.from_numpy(positions).float()
        num_atoms = positions.shape[0]
        center_of_mass = torch.mean(positions, dim=0)
        relative_pos = positions - center_of_mass
        r = torch.norm(positions, dim=1, keepdim=True)
        voltage_feat = torch.full((num_atoms, 1), float(voltage), dtype=torch.float)
        temperature_feat = torch.full((num_atoms, 1), float(temperature), dtype=torch.float)
        time_feat = torch.full((num_atoms, 1), float(time), dtype=torch.float)
        return torch.cat([positions, relative_pos, r, voltage_feat, temperature_feat, time_feat], dim=1)

    def create_data_object(self, coords, voltage, temperature, time, device):
        """Creates a PyG Data object from coordinates and physical conditions."""
        edge_index, edge_attr = self._create_graph_from_coords(coords, device)
        node_features = self._create_node_features(coords, voltage, temperature, time)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# =====================================================================================
# == INFERENCE SCRIPT MAIN LOGIC ==
# =====================================================================================

def save_xyz(filepath, coords, atom_type='Ag', comment="Predicted structure"):
    """Saves coordinates to an XYZ file."""
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    num_atoms = coords.shape[0]
    with open(filepath, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"{comment}\n")
        for i in range(num_atoms):
            f.write(f"{atom_type} {coords[i,0]:.8f} {coords[i,1]:.8f} {coords[i,2]:.8f}\n")
    print(f"Successfully saved predicted structure to {filepath}")

def run_multistep_prediction(args):
    """Main function to run multi-step inference."""
    # --- Setup ---
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from {args.model_path}")
    model = MemristorGNN(
        input_dim=10, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers
    ).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {args.model_path}")
        return
    model.eval()

    # --- Prepare Initial Data ---
    print("Processing initial data...")
    processor = EnhancedAtomicDataProcessor(max_distance=args.max_distance)
    try:
        current_coords = processor._load_xyz_file(args.input_xyz)
    except Exception as e:
        print(f"Failed to load initial XYZ file: {e}. Aborting.")
        return
        
    current_time = args.time
    output_basename, output_ext = os.path.splitext(args.output_xyz)

    # --- Run Inference Loop ---
    print(f"Starting multi-step prediction for {args.num_frames} frames...")
    with torch.no_grad():
        for i in range(1, args.num_frames + 1):
            print(f"--- Predicting frame {i}/{args.num_frames} (Time: {current_time:.2f}) ---")
            
            # 1. Create data object for the current step
            data = processor.create_data_object(
                current_coords, args.voltage, args.temperature, current_time, device
            )
            data = data.to(device)

            # 2. Run the model to get the next state
            predicted_coords = model(data.x, data.edge_index, data.edge_attr)
            
            # 3. Save the predicted frame
            output_path = f"{output_basename}_step_{i}{output_ext}"
            comment = (f"Predicted frame {i} from {os.path.basename(args.input_xyz)}. "
                       f"Time={current_time + args.time_step:.2f}")
            save_xyz(output_path, predicted_coords, comment=comment)

            # 4. Update state for the next iteration
            current_coords = predicted_coords.cpu().numpy()
            current_time += args.time_step

    print("\nMulti-step prediction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multi-step inference with a trained MemristorGNN model.")
    
    # --- File Paths ---
    parser.add_argument('--input_xyz', type=str, required=True, help='Path to the initial input XYZ file.')
    parser.add_argument('--model_path', type=str, default='memristor_physics_model.pth', help='Path to the trained model state dictionary.')
    parser.add_argument('--output_xyz', type=str, default='predicted_structure.xyz', help='Base path for saving the output XYZ files.')
    
    # --- Physical & Simulation Conditions ---
    parser.add_argument('--voltage', type=float, required=True, help='Voltage applied to the system (constant for all steps).')
    parser.add_argument('--temperature', type=float, required=True, help='System temperature (constant for all steps).')
    parser.add_argument('--time', type=float, required=True, help='Initial simulation time of the input file.')
    parser.add_argument('--time_step', type=float, required=True, help='Time increment for each prediction step.')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of future frames to predict.')

    # --- Model Hyperparameters (must match the trained model) ---
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension of the GNN model.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the GNN model.')

    # --- Pre-processing & Device ---
    parser.add_argument('--max_distance', type=float, default=3.5, help='Max distance for radius graph creation.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run the model on.')

    args = parser.parse_args()
    
    # Example Usage from command line:
    # python your_inference_script_name.py --input_xyz path/to/initial.xyz --voltage 1.5 --temperature 300 --time 5000 --time_step 10 --num_frames 5 --output_xyz prediction.xyz

    run_multistep_prediction(args)
