import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

# Try to import GPU-accelerated graph functions
try:
    from torch_cluster import radius_graph, knn_graph
    TORCH_CLUSTER_AVAILABLE = True
    print("torch_cluster is available. Using GPU for graph creation.")
except ImportError:
    TORCH_CLUSTER_AVAILABLE = False
    print("torch_cluster not available. Using CPU fallback for graph creation.")

# ----------------- Enhanced Physics-Based Loss Functions -----------------
class MemristorPhysicsLoss(nn.Module):
    """
    Physics-informed loss function tailored for memristor modeling with:
    - Constrained volume expansion (2% in x,y only, none in z)
    - Fixed electrode layers (top/bottom 3 layers)
    - Anti-collapse penalty focused on filament formation
    - Electric field-aware distance penalties
    """
    def __init__(self,
                 reconstruction_weight=1.0,
                 volume_constraint_weight=10.0,
                 electrode_constraint_weight=50.0,
                 anti_collapse_weight=5.0,
                 field_alignment_weight=2.0,
                 min_distance=2.9,
                 max_xy_expansion=0.02):  # 2% expansion limit
        super(MemristorPhysicsLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.volume_constraint_weight = volume_constraint_weight
        self.electrode_constraint_weight = electrode_constraint_weight
        self.anti_collapse_weight = anti_collapse_weight
        self.field_alignment_weight = field_alignment_weight
        self.min_distance = min_distance
        self.max_xy_expansion = max_xy_expansion
        self.mse_loss = nn.MSELoss()

    def _identify_electrode_atoms(self, coords, batch_vector):
        """
        Identify top and bottom electrode atoms (3 layers each) based on z-coordinate.
        Returns masks for fixed atoms in each graph.
        """
        pred_dense, mask = to_dense_batch(coords, batch_vector)
        batch_size, max_nodes, _ = pred_dense.shape
        
        electrode_masks = []
        for b in range(batch_size):
            valid_coords = pred_dense[b][mask[b]]  # Only valid atoms
            if len(valid_coords) == 0:
                electrode_masks.append(torch.zeros(max_nodes, dtype=torch.bool, device=coords.device))
                continue
                
            z_coords = valid_coords[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_range = z_max - z_min
            
            # Define electrode regions (bottom and top 15% of the structure)
            bottom_threshold = z_min + 0.15 * z_range
            top_threshold = z_max - 0.15 * z_range
            
            # Create mask for electrode atoms
            electrode_mask = torch.zeros(max_nodes, dtype=torch.bool, device=coords.device)
            valid_indices = torch.where(mask[b])[0]
            
            for i, idx in enumerate(valid_indices):
                z = valid_coords[i, 2]
                if z <= bottom_threshold or z >= top_threshold:
                    electrode_mask[idx] = True
                    
            electrode_masks.append(electrode_mask)
        
        return torch.stack(electrode_masks)

    def _calculate_volume_constraint(self, pred_coords, original_coords, batch_vector):
        """
        Penalize volume expansion beyond 2% in x,y directions and any expansion in z.
        """
        pred_dense, mask = to_dense_batch(pred_coords, batch_vector)
        orig_dense, _ = to_dense_batch(original_coords, batch_vector)
        
        volume_loss = 0.0
        batch_size = pred_dense.shape[0]
        
        for b in range(batch_size):
            if not mask[b].any():
                continue
                
            pred_coords_b = pred_dense[b][mask[b]]
            orig_coords_b = orig_dense[b][mask[b]]
            
            # Calculate bounding box dimensions
            pred_min, pred_max = pred_coords_b.min(dim=0)[0], pred_coords_b.max(dim=0)[0]
            orig_min, orig_max = orig_coords_b.min(dim=0)[0], orig_coords_b.max(dim=0)[0]
            
            pred_dims = pred_max - pred_min
            orig_dims = orig_max - orig_min
            
            # Calculate expansion ratios
            expansion_ratios = (pred_dims - orig_dims) / (orig_dims + 1e-8)
            
            # Penalize x,y expansion beyond 2%
            xy_penalty = F.relu(expansion_ratios[:2] - self.max_xy_expansion).sum()
            
            # Penalize any z expansion
            z_penalty = F.relu(expansion_ratios[2])
            
            volume_loss += xy_penalty + 2.0 * z_penalty  # Higher weight for z expansion
        
        return volume_loss / batch_size

    def _calculate_electrode_constraint(self, pred_coords, original_coords, batch_vector):
        """
        Strongly penalize movement of electrode atoms (top/bottom layers).
        """
        electrode_masks = self._identify_electrode_atoms(original_coords, batch_vector)
        
        # Convert to node-level masks
        electrode_node_mask = []
        batch_sizes = torch.bincount(batch_vector)
        start_idx = 0
        
        for b, size in enumerate(batch_sizes):
            batch_electrode_mask = electrode_masks[b][:size]  # Trim to actual size
            electrode_node_mask.append(batch_electrode_mask)
            start_idx += size
        
        electrode_node_mask = torch.cat(electrode_node_mask)
        
        if electrode_node_mask.any():
            electrode_pred = pred_coords[electrode_node_mask]
            electrode_orig = original_coords[electrode_node_mask]
            electrode_loss = self.mse_loss(electrode_pred, electrode_orig)
        else:
            electrode_loss = torch.tensor(0.0, device=pred_coords.device)
            
        return electrode_loss

    def _calculate_field_alignment_loss(self, pred_coords, batch_vector, voltage_features):
        """
        Encourage atomic movements to align with electric field direction.
        For positive voltage, encourage movement towards positive z (top electrode).
        """
        pred_dense, mask = to_dense_batch(pred_coords, batch_vector)
        batch_size = pred_dense.shape[0]
        
        field_loss = 0.0
        for b in range(batch_size):
            if not mask[b].any():
                continue
                
            # Extract voltage for this batch (assuming it's in the node features)
            # You may need to adjust this based on how voltage is stored
            voltage = voltage_features[batch_vector == b][0, -3]  # Assuming voltage is 3rd from last
            
            if abs(voltage) < 1e-6:  # Skip if no voltage
                continue
                
            coords_b = pred_dense[b][mask[b]]
            z_coords = coords_b[:, 2]
            
            # Calculate center of mass movement in z-direction
            z_com = z_coords.mean()
            z_range = z_coords.max() - z_coords.min()
            z_com_normalized = (z_com - z_coords.min()) / (z_range + 1e-8)  # 0 to 1
            
            # For positive voltage, encourage movement towards top (z=1)
            # For negative voltage, encourage movement towards bottom (z=0)
            if voltage > 0:
                target_z_com = 0.6  # Slight bias towards top
            else:
                target_z_com = 0.4  # Slight bias towards bottom
                
            field_loss += F.mse_loss(z_com_normalized, torch.tensor(target_z_com, device=pred_coords.device))
        
        return field_loss / batch_size

    def forward(self, pred_coords, true_coords, original_coords, batch_vector, node_features):
        """
        Calculate the combined memristor-specific physics-informed loss.
        """
        # 1. Reconstruction Loss (reduced weight since it was problematic)
        recon_loss = self.mse_loss(pred_coords, true_coords)

        # 2. Volume Constraint Loss
        volume_loss = self._calculate_volume_constraint(pred_coords, original_coords, batch_vector)

        # 3. Electrode Constraint Loss (keep electrode layers fixed)
        electrode_loss = self._calculate_electrode_constraint(pred_coords, original_coords, batch_vector)

        # 4. Anti-Collapse Loss (focused on realistic atomic distances)
        pred_dense, mask = to_dense_batch(pred_coords, batch_vector)
        batch_size, max_nodes, _ = pred_dense.shape
        
        collapse_loss = 0.0
        for b in range(batch_size):
            if not mask[b].any():
                continue
                
            coords_b = pred_dense[b][mask[b]]
            if len(coords_b) < 2:
                continue
                
            # Calculate pairwise distances
            distances = torch.cdist(coords_b, coords_b, p=2)
            
            # Exclude self-distances and focus on nearest neighbors
            eye_mask = ~torch.eye(len(coords_b), dtype=torch.bool, device=pred_coords.device)
            valid_distances = distances[eye_mask]
            
            # Penalize distances below minimum threshold
            violations = F.relu(self.min_distance - valid_distances)
            collapse_loss += torch.mean(violations ** 2)  # Quadratic penalty
        
        collapse_loss /= batch_size

        # 5. Electric Field Alignment Loss
        field_loss = self._calculate_field_alignment_loss(pred_coords, batch_vector, node_features)

        # 6. Combine all losses
        total_loss = (self.reconstruction_weight * recon_loss +
                      self.volume_constraint_weight * volume_loss +
                      self.electrode_constraint_weight * electrode_loss +
                      self.anti_collapse_weight * collapse_loss +
                      self.field_alignment_weight * field_loss)

        return total_loss, recon_loss, volume_loss, electrode_loss, collapse_loss, field_loss

# ----------------- Enhanced GNN Model for Memristor -----------------
class MemristorGNN(nn.Module):
    """
    Enhanced GNN model specifically designed for memristor filament formation.
    Includes voltage-aware processing and constraint-aware predictions.
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

# ----------------- Keep existing AtomicDataset and EnhancedAtomicDataProcessor classes unchanged -----------------
class AtomicDataset(Dataset):
    """
    A PyTorch Geometric Dataset for loading pre-processed atomic graph data.
    It assumes a sequence of files, where each file represents a single data point (graph).
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(AtomicDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            return []
        try:
            return sorted([f for f in os.listdir(self.processed_dir) if f.startswith('data_') and f.endswith('.pt')],
                          key=lambda f: int(f.split('_')[1].split('.')[0]))
        except (ValueError, IndexError):
             return sorted([f for f in os.listdir(self.processed_dir) if f.startswith('data_') and f.endswith('.pt')])

    def download(self):
        pass

    def process(self):
        print(f"Dataset 'process' called. Please place processed files in '{self.processed_dir}' or run the external pre-processing script.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

class EnhancedAtomicDataProcessor:
    """
    Handles the pre-processing of raw LAMMPS data into PyG Data objects.
    """
    def __init__(self, k_neighbors=8, max_distance=3.5):
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

    def _load_xyz_file(self, filepath):
        try:
            return pd.read_csv(filepath, sep='\s+', header=None, engine='python').values.astype(np.float32)
        except (ValueError, pd.errors.ParserError):
            print(f"Warning: Whitespace delimiter failed for {filepath}. Trying comma.")
            return pd.read_csv(filepath, sep=',', header=None).values.astype(np.float32)

    def _create_graph_from_coords(self, coords):
        if not torch.is_tensor(coords):
            coords = torch.from_numpy(coords).float()

        device = 'cuda' if torch.cuda.is_available() and TORCH_CLUSTER_AVAILABLE else 'cpu'
        coords = coords.to(device)

        edge_index = radius_graph(coords, r=self.max_distance, loop=False, max_num_neighbors=32)
        
        if edge_index.numel() == 0:
            print("Warning: No neighbors found in radius search. Falling back to 1-NN.")
            edge_index = knn_graph(coords, k=1, loop=False)

        row, col = edge_index
        edge_vectors = coords[row] - coords[col]
        edge_attr = torch.norm(edge_vectors, p=2, dim=1).unsqueeze(1)

        return edge_index.to('cpu'), edge_attr.to('cpu')

    def _create_node_features(self, positions, voltage, temperature, time):
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

    def _find_data_paths(self, base_dir):
        xyz_candidates = ["xyz_files", "output_mesh_xyz/xyz_files", "output_mesh_xyz"]
        xyz_dir = None
        for cand in xyz_candidates:
            path = os.path.join(base_dir, cand)
            if os.path.isdir(path):
                xyz_dir = path
                break
        
        meta_candidates = ["metadata.csv", "voltage_temperature_data.csv"]
        metadata_file = None
        for cand in meta_candidates:
            path = os.path.join(base_dir, cand)
            if os.path.isfile(path):
                metadata_file = path
                break
                
        return xyz_dir, metadata_file

    def process_and_save(self, lammps_dirs, output_path, seq_step=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        graph_counter = 0
        for base_dir in lammps_dirs:
            print(f"\nProcessing base directory: {base_dir}")
            xyz_dir, metadata_file = self._find_data_paths(base_dir)

            if not xyz_dir or not metadata_file:
                print(f"Warning: Skipping directory {base_dir}. Could not find required xyz and/or metadata files.")
                continue
            
            print(f"  Found xyz data in: {xyz_dir}")
            print(f"  Found metadata in: {metadata_file}")

            try:
                metadata_df = pd.read_csv(metadata_file)
                metadata_df = metadata_df.sort_values(by='time').reset_index(drop=True)
            except Exception as e:
                print(f"Error loading metadata for {base_dir}: {e}")
                continue
            
            for i in tqdm(range(len(metadata_df) - seq_step), desc=f"Creating graphs for {os.path.basename(base_dir)}"):
                input_meta = metadata_df.iloc[i]
                target_meta = metadata_df.iloc[i + seq_step]

                try:
                    input_coords = self._load_xyz_file(os.path.join(xyz_dir, input_meta['filename']))
                    target_coords = self._load_xyz_file(os.path.join(xyz_dir, target_meta['filename']))
                    
                    if input_coords.shape[0] != target_coords.shape[0]:
                        print(f"Warning: Mismatch atom count for {input_meta['filename']}. Skipping.")
                        continue
                        
                    edge_index, edge_attr = self._create_graph_from_coords(input_coords)
                    node_features = self._create_node_features(
                        input_coords, input_meta['volt_curr'], input_meta['temp_curr'], input_meta['time']
                    )

                    data = Data(
                        x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=torch.from_numpy(target_coords).float(),
                        original_pos=torch.from_numpy(input_coords).float()
                    )

                    torch.save(data, os.path.join(output_path, f'data_{graph_counter}.pt'))
                    graph_counter += 1

                except Exception as e:
                    print(f"\nError processing file pair starting with {input_meta['filename']}: {e}")
        
        print(f"\nPre-processing complete. Saved {graph_counter} graphs to {output_path}.")

# ----------------- Enhanced Training Functions -----------------
def train_model(model, train_loader, optimizer, physics_criterion, device, epoch):
    model.train()
    total_loss = 0
    total_losses = {'recon': 0, 'volume': 0, 'electrode': 0, 'collapse': 0, 'field': 0}
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred_coords = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        loss, recon, volume, electrode, collapse, field = physics_criterion(
            pred_coords, batch.y, batch.original_pos, batch.batch, batch.x
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_losses['recon'] += recon.item()
        total_losses['volume'] += volume.item()
        total_losses['electrode'] += electrode.item()
        total_losses['collapse'] += collapse.item()
        total_losses['field'] += field.item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Vol': f'{volume.item():.4f}',
            'Elec': f'{electrode.item():.4f}',
            'Coll': f'{collapse.item():.4f}'
        })
        
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    
    print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.6f}")
    for key, val in total_losses.items():
        print(f"  {key.capitalize()}: {val/num_batches:.6f}")
    
    return avg_loss

def validate_model(model, val_loader, physics_criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device)
            pred_coords = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            loss, _, _, _, _, _ = physics_criterion(
                pred_coords, batch.y, batch.original_pos, batch.batch, batch.x
            )
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss

# ----------------- Main Training Script -----------------
def main():
    # --- Configuration ---
    lammps_dirs = [
        r"D:\prakhar\sprint00\simul1", 
        r"D:\prakhar\sprint00\simul2",
        r"D:\prakhar\sprint00\simul3",
        r"D:\prakhar\sprint00\simul4",
        r"D:\prakhar\sprint00\simul5"
    ]
    dataset_root = r"D:\prakhar\sprint00\enhanced_graph_dataset"
    
    # Model and Training Parameters
    epochs = 100
    batch_size = 6
    hidden_dim = 32
    num_layers = 4
    lr = 1e-3  # Reduced learning rate for more stable training
    save_model_path = "memristor_physics_model.pth"
    
    # Enhanced physics parameters for memristor
    physics_params = {
        'reconstruction_weight': 0.1,  # Reduced as it was problematic
        'volume_constraint_weight': 10.0,  # High weight for volume constraint
        'electrode_constraint_weight': 50.0,  # Very high for fixed electrodes
        'anti_collapse_weight': 5.0,  # Keep collapse prevention
        'field_alignment_weight': 2.0,  # Encourage field-aligned movement
        'min_distance': 2.9,
        'max_xy_expansion': 0.02  # 2% expansion limit
    }

    # --- 1. Pre-processing Step ---
    processor = EnhancedAtomicDataProcessor(k_neighbors=8, max_distance=3.5)
    processed_dir = os.path.join(dataset_root, 'processed')
    
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        print("Processed data not found. Starting pre-processing...")
        processor.process_and_save(lammps_dirs, processed_dir, seq_step=1)
    else:
        print("Found existing processed data. Skipping pre-processing.")

    # --- 2. Dataset and DataLoader Setup ---
    dataset = AtomicDataset(root=dataset_root)
    
    if len(dataset) == 0:
        print("\n" + "="*50)
        print("ERROR: Dataset is empty.")
        print("="*50)
        return

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Dataset size: {len(dataset)}. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # --- 3. Model, Loss, and Optimizer Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MemristorGNN(
        input_dim=10, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)
    
    physics_criterion = MemristorPhysicsLoss(**physics_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=5, verbose=True)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, physics_criterion, device, epoch)
        val_loss = validate_model(model, val_loader, physics_criterion, device)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)
            print(f"--- New best model saved to {save_model_path} with validation loss: {val_loss:.6f} ---")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    main()