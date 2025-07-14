import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from torch_geometric.nn import NNConv
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
class MemristorPhysicsLoss(torch.nn.Module):
    """
    Physics-informed loss function tailored for memristor modeling with:
    - Constrained volume expansion (2% in x,y only, none in z)
    - Fixed electrode layers (top/bottom 3 layers)
    - Anti-collapse penalty focused on filament formation
    - Removed field alignment loss for cleaner training
    """
    def __init__(self,
                 reconstruction_weight=1.0,
                 volume_constraint_weight=10.0,
                 electrode_constraint_weight=50.0,
                 anti_collapse_weight=5.0,
                 
                 min_distance=2.9,
                 max_xy_expansion=0.02,
                 z_penalty_weight=50,
                 huber_delta=0.5):  # 2% expansion limit
        super(MemristorPhysicsLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.volume_constraint_weight = volume_constraint_weight
        self.electrode_constraint_weight = electrode_constraint_weight
        self.anti_collapse_weight = anti_collapse_weight
        
        self.min_distance = min_distance
        self.max_xy_expansion = max_xy_expansion
        self.mse_loss = torch.nn.MSELoss()
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta)
        self.z_penalty_weight = z_penalty_weight
    
    def _identify_electrode_atoms(self, coords, batch_vector, electrode_fraction=0.1):
        """
        Identify top and bottom electrode atoms based on z-coordinate.
        
        Args:
            coords: Tensor of atomic coordinates [N, 3]
            batch_vector: Batch assignment for each atom
            electrode_fraction: Fraction of structure height to consider as electrode (default: 0.15)
        
        Returns:
            List of boolean masks for electrode atoms in each graph
        """
        pred_dense, mask = to_dense_batch(coords, batch_vector)
        batch_size, max_nodes, _ = pred_dense.shape
        
        electrode_masks = []
        
        for b in range(batch_size):
            # Get valid coordinates for this batch
            valid_mask = mask[b]
            valid_coords = pred_dense[b][valid_mask]
            
            # Handle empty batch
            if len(valid_coords) == 0:
                electrode_masks.append(torch.zeros(max_nodes, dtype=torch.bool, device=coords.device))
                continue
            
            # Extract z-coordinates and compute range
            z_coords = valid_coords[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_range = z_max - z_min
            
            # Handle flat structures (no z variation)
            if z_range < 1e-6:  # Add small tolerance for numerical precision
                electrode_masks.append(torch.zeros(max_nodes, dtype=torch.bool, device=coords.device))
                continue
            
            # Define electrode regions
            bottom_threshold = z_min + electrode_fraction * z_range
            top_threshold = z_max - electrode_fraction * z_range
            
            # Vectorized approach for better performance
            electrode_mask = torch.zeros(max_nodes, dtype=torch.bool, device=coords.device)
            valid_indices = torch.where(valid_mask)[0]
            
            # Check which atoms are in electrode regions
            is_electrode = (z_coords <= bottom_threshold) | (z_coords >= top_threshold)
            electrode_mask[valid_indices] = is_electrode
            
            electrode_masks.append(electrode_mask)
        
        return electrode_masks

    def _identify_filament_atoms(self, coords, batch_vector, electrode_fraction=0.405, filament_fraction=0.19):
        """
        Identify filament atoms (middle region between electrodes).
        
        Args:
            coords: Tensor of atomic coordinates [N, 3]
            batch_vector: Batch assignment for each atom
            electrode_fraction: Fraction from top/bottom to exclude as electrodes
            filament_fraction: Fraction of middle region to consider as filament zone
        
        Returns:
            List of boolean masks for filament atoms in each graph
        """
        pred_dense, mask = to_dense_batch(coords, batch_vector)
        batch_size, max_nodes, _ = pred_dense.shape
        
        filament_masks = []
        
        for b in range(batch_size):
            # Get valid coordinates for this batch
            valid_mask = mask[b]
            valid_coords = pred_dense[b][valid_mask]
            
            # Handle empty batch
            if len(valid_coords) == 0:
                filament_masks.append(torch.zeros(max_nodes, dtype=torch.bool, device=coords.device))
                continue
            
            # Extract z-coordinates and compute range
            z_coords = valid_coords[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_range = z_max - z_min
                               
            # Define electrode regions (same as your existing method)
            bottom_electrode_threshold = z_min + electrode_fraction * z_range
            top_electrode_threshold = z_max - electrode_fraction * z_range
            
            # Define filament region (middle part, excluding electrodes)
            middle_z = (z_min + z_max) / 2
            filament_half_height = (filament_fraction * (top_electrode_threshold - bottom_electrode_threshold)) / 2
            filament_bottom = middle_z - filament_half_height
            filament_top = middle_z + filament_half_height
            
            # Create filament mask
            filament_mask = torch.zeros(max_nodes, dtype=torch.bool, device=coords.device)
            valid_indices = torch.where(valid_mask)[0]
            
            # Atoms are filament if they're in the middle region and not in electrode regions
            is_in_middle = (z_coords >= bottom_electrode_threshold) & (z_coords <= top_electrode_threshold)
            is_in_filament_zone = (z_coords >= filament_bottom) & (z_coords <= filament_top)
            is_filament = is_in_middle & is_in_filament_zone
            
            filament_mask[valid_indices] = is_filament
            filament_masks.append(filament_mask)
        
        return filament_masks
    
    def get_filament_atom_counts(self, coords, batch_vector, electrode_fraction=0.405, filament_fraction=0.19):
    
        filament_masks = self._identify_filament_atoms(coords, batch_vector, electrode_fraction, filament_fraction)
        return [mask.sum().item() for mask in filament_masks]
    '''
    def diffusion_loss(self, pred_coords, true_coords, original_coords,batch_vector):
        """
        physics loss based on ion diffusion from Jart VCM
        """
        true_filament_atoms = self.get_filament_atom_counts(true_coords,batch_vector)
        pred_filament_atoms = self.get_filament_atom_counts(pred_coords,batch_vector)
        original_filament_atoms = self.get_filament_atom_counts(original_coords,batch_vector)
    '''
        


    def _calculate_weighted_reconstruction_loss(self, pred_coords, true_coords, batch_vector, 
                                            filament_weight=50.0, electrode_weight=1.0):
        """
        Calculate reconstruction loss with heavy weighting toward filament atoms.
        
        Args:
            pred_coords: Predicted coordinates
            true_coords: Target coordinates  
            batch_vector: Batch assignment for each atom
            filament_weight: Weight multiplier for filament atom loss
            electrode_weight: Weight multiplier for electrode atom loss
        
        Returns:
            Weighted reconstruction loss
        """
        # Identify filament atoms
        filament_masks = self._identify_filament_atoms(true_coords, batch_vector)
        
        # Convert to node-level masks
        batch_sizes = torch.bincount(batch_vector)
        start_idx = 0
        
        total_filament_loss = 0.0
        total_electrode_loss = 0.0
        filament_count = 0
        electrode_count = 0
        
        for b, size in enumerate(batch_sizes):
            end_idx = start_idx + size
            batch_filament_mask = filament_masks[b][:size]
            
            batch_pred = pred_coords[start_idx:end_idx]
            batch_true = true_coords[start_idx:end_idx]
            
            # Filament atoms loss
            if batch_filament_mask.any():
                filament_pred = batch_pred[batch_filament_mask]
                filament_true = batch_true[batch_filament_mask]
                filament_loss = self.huber_loss(filament_pred, filament_true)
                total_filament_loss += filament_loss
                filament_count += batch_filament_mask.sum().item()
            
            # Electrode atoms loss (everything not filament)
            electrode_mask = ~batch_filament_mask
            if electrode_mask.any():
                electrode_pred = batch_pred[electrode_mask]
                electrode_true = batch_true[electrode_mask]
                electrode_loss = self.mse_loss(electrode_pred, electrode_true)
                total_electrode_loss += electrode_loss
                electrode_count += electrode_mask.sum().item()
            
            start_idx = end_idx
        
        # Normalize by number of atoms in each category
        if filament_count > 0:
            avg_filament_loss = total_filament_loss / len(batch_sizes)
        else:
            avg_filament_loss = torch.tensor(0.0, device=pred_coords.device)
        
        if electrode_count > 0:
            avg_electrode_loss = total_electrode_loss / len(batch_sizes)
        else:
            avg_electrode_loss = torch.tensor(0.0, device=pred_coords.device)
        
        # Apply weights and combine
        weighted_loss = (filament_weight * avg_filament_loss + 
                        electrode_weight * avg_electrode_loss)
        
        return weighted_loss, avg_filament_loss, avg_electrode_loss

    def _calculate_volume_constraint(self, pred_coords, original_coords, batch_vector):
        """
        Penalize volume expansion beyond 2% in x,y directions and any change in z.
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
            
            # Penalize any change in z (expansion or contraction)
            z_penalty = torch.abs(expansion_ratios[2])
            
            volume_loss += xy_penalty + self.z_penalty_weight * z_penalty
        
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

    
    def _calculate_collapse_loss(self, pred_coords, batch_vector):
        """
        Calculate anti-collapse loss - only penalize when atoms get closer than min_distance.
        """
        pred_dense, mask = to_dense_batch(pred_coords, batch_vector)
        batch_size, max_nodes, _ = pred_dense.shape
        
        collapse_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            if not mask[b].any():
                continue
                
            coords_b = pred_dense[b][mask[b]]
            if len(coords_b) < 2:
                continue
                
            valid_batches += 1
            
            # Calculate pairwise distances efficiently
            distances = torch.cdist(coords_b, coords_b, p=2)
            
            # Create mask to exclude self-distances (diagonal)
            n_atoms = len(coords_b)
            non_diagonal_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=pred_coords.device)
            
            # Get valid pairwise distances (excluding self-distances)
            valid_distances = distances[non_diagonal_mask]
            
            # Only penalize distances that are below the minimum threshold
            # This prevents atoms from getting too close to each other
            violations = valid_distances[valid_distances < self.min_distance]
            
            if len(violations) > 0:
                # Quadratic penalty for violations - stronger penalty for closer atoms
                penalty = torch.sum((self.min_distance - violations) ** 2)
                collapse_loss += penalty
        
        # Average over valid batches
        if valid_batches > 0:
            collapse_loss /= valid_batches
        
        return collapse_loss
           
    def forward(self, pred_coords, true_coords, original_coords, batch_vector):
        """
        Calculate the combined memristor-specific physics-informed loss.
        """
        # 1. Reconstruction Loss
        #recon_loss = self.huber_loss(pred_coords, true_coords)

        weighted_recon_loss, filament_loss, electrode_loss = self._calculate_weighted_reconstruction_loss(
        pred_coords, true_coords, batch_vector)

        # 2. Volume Constraint Loss
        volume_loss = self._calculate_volume_constraint(pred_coords, original_coords, batch_vector)

        # 3. Electrode Constraint Loss (keep electrode layers fixed)
        #electrode_loss = self._calculate_electrode_constraint(pred_coords, original_coords, batch_vector)

        # 4. Anti-Collapse Loss (focused on realistic atomic distances)
        collapse_loss = self._calculate_collapse_loss(pred_coords, batch_vector)

    
        

        # 6. Combine all losses
        total_loss = (self.reconstruction_weight * weighted_recon_loss +
                      self.volume_constraint_weight * volume_loss +
                      self.electrode_constraint_weight * electrode_loss +
                      self.anti_collapse_weight * collapse_loss 
                      )/(self.reconstruction_weight+
                        self.volume_constraint_weight+
                        self.electrode_constraint_weight+
                        self.anti_collapse_weight)
                                                                      

        return total_loss, weighted_recon_loss, volume_loss, electrode_loss, collapse_loss


# ----------------- Enhanced GNN Model for Cycle Simulation -----------------
class MemristorCycleGNN(nn.Module):
    """
    Enhanced GNN model for memristor cycle simulation that takes:
    - Current state (xyz, volt, temp, time)
    - Target conditions (volt, temp, time)
    And predicts the target xyz coordinates.
    """
    def __init__(self, input_dim=3, edge_dim=1, hidden_dim=64, output_dim=3,
                 num_layers=6, dropout=0.1, condition_embedding_dim=4):
        super(MemristorCycleGNN, self).__init__()
        self.dropout = dropout
        self.condition_embedding_dim = condition_embedding_dim
        self.hidden_dim = hidden_dim

        self.current_volt_encoder = nn.Sequential(
            nn.Linear(1, condition_embedding_dim),
            nn.ReLU(),
            nn.Linear(condition_embedding_dim, condition_embedding_dim),
            nn.LayerNorm(condition_embedding_dim)
        )
        
        self.current_temp_encoder = nn.Sequential(
            nn.Linear(1, condition_embedding_dim),
            nn.ReLU(),
            nn.Linear(condition_embedding_dim, condition_embedding_dim),
            nn.LayerNorm(condition_embedding_dim)
        )
        
        self.target_volt_encoder = nn.Sequential(
            nn.Linear(1, condition_embedding_dim),
            nn.ReLU(),
            nn.Linear(condition_embedding_dim, condition_embedding_dim),
            nn.LayerNorm(condition_embedding_dim)
        )
        
        self.target_temp_encoder = nn.Sequential(
            nn.Linear(1, condition_embedding_dim),
            nn.ReLU(),
            nn.Linear(condition_embedding_dim, condition_embedding_dim),
            nn.LayerNorm(condition_embedding_dim)
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(2, condition_embedding_dim), 
            nn.ReLU(),
            nn.Linear(condition_embedding_dim, condition_embedding_dim),
            nn.LayerNorm(condition_embedding_dim)
        )

        # Condition fusion network
        condition_dim = 5 * condition_embedding_dim  
        self.condition_fusion = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        enhanced_input_dim = input_dim + hidden_dim  
        
        self.nn_input = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * enhanced_input_dim),
            nn.Dropout(dropout)
        )
        self.input_conv = NNConv(enhanced_input_dim, hidden_dim, self.nn_input, aggr='mean')
        self.input_bn = nn.LayerNorm(hidden_dim)

        self.hidden_convs = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                nn.Dropout(dropout)
            )
            self.hidden_convs.append(NNConv(hidden_dim, hidden_dim, nn_hidden, aggr='mean'))
            self.hidden_bns.append(nn.LayerNorm(hidden_dim))
            
            # Simple self-attention for better feature interaction
            self.attention_layers.append(nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True))

        # Output layer with condition-aware prediction
        self.nn_output = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            nn.Dropout(dropout)
        )
        self.output_conv = NNConv(hidden_dim, hidden_dim, self.nn_output, aggr='mean')
        
        coord_predictor_input_dim = hidden_dim + (5 * condition_embedding_dim)

        self.coord_predictor = nn.Sequential(
            nn.Linear(coord_predictor_input_dim, hidden_dim), # 1st layer
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2), # 2nd layer
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4), # **NEW: 3rd layer**
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, hidden_dim // 8), # **NEW: 4th layer** (even deeper, potentially smaller)
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 8, output_dim) # Final output layer
        )

    def forward(self, x, edge_index, edge_attr, current_conditions, target_conditions, batch=None):
        """
        Forward pass for cycle simulation.
        
        Args:
            x: Node features [N, 3] (xyz coordinates)
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge attributes [E, 1]
            current_conditions: [N, 3] (current_volt, current_temp, current_time)
            target_conditions: [N, 3] (target_volt, target_temp, target_time)
            batch: Batch vector for batched graphs
        """
        # Store original coordinates
        original_coords = x
        
        # Extract condition information
        current_volt = current_conditions[:, 0:1]
        current_temp = current_conditions[:, 1:2]
        current_time = current_conditions[:, 2:3]
        
        target_volt = target_conditions[:, 0:1]
        target_temp = target_conditions[:, 1:2]
        target_time = target_conditions[:, 2:3]
        
        # Create condition embeddings
        current_volt_emb = self.current_volt_encoder(current_volt)
        current_temp_emb = self.current_temp_encoder(current_temp)
        target_volt_emb = self.target_volt_encoder(target_volt)
        target_temp_emb = self.target_temp_encoder(target_temp)
        
        # Time embedding
        time_features = torch.cat([current_time, target_time], dim=1)
        time_emb = self.time_encoder(time_features)
        
        # Fuse all condition embeddings
        condition_features = torch.cat([
            current_volt_emb, current_temp_emb, 
            target_volt_emb, target_temp_emb, 
            time_emb
        ], dim=1)
        
        fused_conditions = self.condition_fusion(condition_features)
        
        # Enhance input features with condition context
        enhanced_x = torch.cat([original_coords, fused_conditions], dim=1)

        # Input layer
        x_processed = self.input_conv(enhanced_x, edge_index, edge_attr)
        x_processed = self.input_bn(x_processed)
        x_processed = F.relu(x_processed)
        x_processed = F.dropout(x_processed, p=self.dropout, training=self.training)

        # Hidden layers with residual connections and attention
        for conv, bn, attn in zip(self.hidden_convs, self.hidden_bns, self.attention_layers):
            residual = x_processed
            
            # Graph convolution
            x_processed = conv(x_processed, edge_index, edge_attr)
            x_processed = bn(x_processed)
            x_processed = F.relu(x_processed)
            
            # Self-attention (convert to batch format)
            if batch is not None:
                x_dense, mask = to_dense_batch(x_processed, batch)
                batch_size, max_nodes, feat_dim = x_dense.shape
                
                # Apply attention
                x_dense_flat = x_dense.view(-1, max_nodes, feat_dim)
                attn_out, _ = attn(x_dense_flat, x_dense_flat, x_dense_flat, key_padding_mask=~mask)
                
                # Convert back to node format
                x_processed = attn_out[mask]
            
            x_processed = F.dropout(x_processed, p=self.dropout, training=self.training)
            x_processed = x_processed + residual  # Residual connection

        # Output layer
        x_processed = self.output_conv(x_processed, edge_index, edge_attr)
        
        # Final coordinate prediction with condition context
        final_features = torch.cat([x_processed, condition_features], dim=1)
        # The model now directly predicts the displacement (delta)
        coordinate_delta = self.coord_predictor(final_features)
        
        # Final prediction: add predicted displacement to original coordinates
        output_coords = original_coords + coordinate_delta
        
        return output_coords

# ----------------- Enhanced Data Processing for Cycle Simulation -----------------
class CycleAtomicDataProcessor:
    """
    Handles the pre-processing of raw L data for cycle simulation.
    Creates training pairs with current state + target conditions -> target state
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

    def _find_data_paths(self, base_dir):
        xyz_candidates = ["xyz_files"]
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

    def process_and_save(self, lammps_dirs, output_path, seq_step=50, max_time_gap=500.0):
        """
        Process data for cycle simulation with enhanced sampling strategy.
        Creates input-target graph pairs for molecular dynamics prediction.
        """
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
            
            # Enhanced sampling: create multiple target pairs per input
            for i in tqdm(range(len(metadata_df) - seq_step), desc=f"Creating cycle graphs for {os.path.basename(base_dir)}"):
                input_meta = metadata_df.iloc[i]
                
                # Create multiple targets with different time steps
                for step in [seq_step]:
                    if i + step >= len(metadata_df):
                        break
                        
                    target_meta = metadata_df.iloc[i + step]
                    time_gap = target_meta['time'] - input_meta['time']
                    
                    # Skip if time gap is too large
                    if time_gap > max_time_gap:
                        continue

                    # Debug: Ensure we're using different files
                    if input_meta['filename'] == target_meta['filename']:
                        print(f"Warning: Same file used for input and target at index {i}. Skipping.")
                        continue

                    try:
                        input_coords = self._load_xyz_file(os.path.join(xyz_dir, input_meta['filename']))
                        target_coords = self._load_xyz_file(os.path.join(xyz_dir, target_meta['filename']))
                        
                        if input_coords.shape[0] != target_coords.shape[0]:
                            print(f"Warning: Mismatch atom count for {input_meta['filename']}. Skipping.")
                            continue
                        
                        # Create separate graph structures for input and target
                        input_edge_index, input_edge_attr = self._create_graph_from_coords(input_coords)
                        target_edge_index, target_edge_attr = self._create_graph_from_coords(target_coords)
                        
                        # Prepare current and target conditions
                        num_atoms = input_coords.shape[0]
                        current_conditions = np.column_stack([
                            np.full(num_atoms, input_meta['volt_curr']),
                            np.full(num_atoms, input_meta['temp_curr']),
                            np.full(num_atoms, input_meta['time'])
                        ])
                        
                        target_conditions = np.column_stack([
                            np.full(num_atoms, target_meta['volt_curr']),
                            np.full(num_atoms, target_meta['temp_curr']),
                            np.full(num_atoms, target_meta['time'])
                        ])

                        # Create data object with both input and target graph structures
                        data = Data(
                            # Input graph
                            x=torch.from_numpy(input_coords).float(),
                            edge_index=input_edge_index,
                            edge_attr=input_edge_attr,
                            
                            # Target graph
                            target_x=torch.from_numpy(target_coords).float(),
                            target_edge_index=target_edge_index,
                            target_edge_attr=target_edge_attr,
                            
                            # Conditions
                            current_conditions=torch.from_numpy(current_conditions).float(),
                            target_conditions=torch.from_numpy(target_conditions).float(),
                            
                            # Keep original format for backward compatibility
                            y=torch.from_numpy(target_coords).float(),
                            original_pos=torch.from_numpy(input_coords).float(),
                            
                            # Additional metadata for debugging
                            input_filename=input_meta['filename'],
                            target_filename=target_meta['filename'],
                            time_gap=time_gap
                        )

                        torch.save(data, os.path.join(output_path, f'data_{graph_counter}.pt'))
                        graph_counter += 1

                    except Exception as e:
                        print(f"\nError processing file pair starting with {input_meta['filename']}: {e}")
            
        print(f"\nPre-processing complete. Saved {graph_counter} graphs to {output_path}.")
    
class AtomicDataset(Dataset):
    """
    A PyTorch Geometric Dataset for loading pre-processed atomic graph data.
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

# ----------------- Enhanced Training Functions -----------------
def train_model(model, train_loader, optimizer, physics_criterion, device, epoch):
    model.train()
    total_loss = 0
    total_losses = {'recon': 0, 'volume': 0, 'electrode': 0, 'collapse': 0}
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred_coords = model(
            batch.x, batch.edge_index, batch.edge_attr, 
            batch.current_conditions, batch.target_conditions, batch.batch
        )
        
        loss, recon, volume, electrode, collapse= physics_criterion(
            pred_coords, batch.y, batch.original_pos, batch.batch
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_losses['recon'] += recon.item()
        total_losses['volume'] += volume.item()
        total_losses['electrode'] += electrode.item()
        total_losses['collapse'] += collapse.item()
        
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{recon.item():.4f}',
            'Vol': f'{volume.item():.4f}',
            
            'coll': f'{collapse.item():.4f}'
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
    
    if not val_loader: # Handle case where val_loader is empty
        print("Validation loader is empty, skipping validation.")
        return float('inf')

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device)
            pred_coords = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.current_conditions, batch.target_conditions, batch.batch
            )
            
            loss, _, _, _, _, _ = physics_criterion(
                pred_coords, batch.y, batch.original_pos, batch.batch
            )
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss

# ----------------- Main Training Script -----------------
def main():
    # --- Configuration ---
    # IMPORTANT: Make sure these paths are correct for your system
    lammps_dirs = [
        r"D:\prakhar\sprint01\simul1",
        #r"D:\prakhar\sprint01\simul2", 
        #r"D:\prakhar\sprint01\simul3",
        #r"D:\prakhar\sprint01\simul4",
        #r"D:\prakhar\sprint01\simul5"
    ]
    dataset_root = r"D:\prakhar\sprint01\cycle_graph_dataset"
    
    # Model and Training Parameters
    epochs = 1000             
    batch_size = 1
    hidden_dim = 32
    num_layers = 5
    lr = 0.001             
    dropout = 0.2
    val_split = 0.0       

    # Model loading configuration
    resume_training = True  # Set to True to load existing model, False to start fresh
    model_path = 'best_model.pt'  # Path to the saved model
    
    # --- 1. Pre-processing ---
    processed_dir = os.path.join(dataset_root, 'processed')
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        print(f"Processed data not found in '{processed_dir}'.")
        print("Starting pre-processing of LAMMPS data...")
        processor = CycleAtomicDataProcessor(k_neighbors=128, max_distance=25)
        processor.process_and_save(lammps_dirs, processed_dir, seq_step=100, max_time_gap=2000.0)
    else:
        print(f"Found pre-processed data in '{processed_dir}'. Skipping pre-processing.")

    # --- 2. Dataset and DataLoader Setup ---
    print("\nLoading dataset...")
    dataset = AtomicDataset(root=dataset_root)
    print(f"Dataset loaded successfully with {len(dataset)} graphs.")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Please check the pre-processing step and data paths.")
        return

    # Create a random split for training and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(val_split * dataset_size))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # Handle the case of an empty validation set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None

    # --- 3. Model, Criterion, and Optimizer Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = MemristorCycleGNN(
        input_dim=3,
        edge_dim=1,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # --- 4. Load existing model if requested ---
    start_epoch = 1
    best_train_loss = float('inf')
    
    if resume_training and os.path.exists(model_path):
        try:
            print(f"\nLoading existing model from '{model_path}'...")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Full checkpoint with metadata
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                    if 'best_loss' in checkpoint:
                        best_train_loss = checkpoint['best_loss']
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                    print(f"Previous best loss: {checkpoint.get('best_loss', 'unknown')}")
                elif 'state_dict' in checkpoint:
                    # Alternative checkpoint format
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict is the state_dict itself
                    model.load_state_dict(checkpoint)
            else:
                # Direct state_dict
                model.load_state_dict(checkpoint)
            
            print("Model loaded successfully! Resuming training...")
            
        except Exception as e:
            print(f"WARNING: Failed to load model from '{model_path}': {e}")
            print("Starting training with a fresh model...")
            start_epoch = 1
            best_train_loss = float('inf')
    elif resume_training:
        print(f"WARNING: Model file '{model_path}' not found. Starting training with a fresh model...")
    else:
        print("Starting training with a fresh model...")

    print("\nModel Architecture Initialized.")

    physics_criterion = MemristorPhysicsLoss(
        reconstruction_weight=100.0,
        volume_constraint_weight=0.0,
        electrode_constraint_weight=0.0,
        anti_collapse_weight=0.0,
       
        min_distance=2.9,
        max_xy_expansion=0.1,
        z_penalty_weight=5,
        huber_delta=1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # --- 5. Training Loop ---
    best_model_path = 'best_model.pt'

    print(f"\n--- Starting Training from Epoch {start_epoch} ---")
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        train_loss = train_model(model, train_loader, optimizer, physics_criterion, device, epoch)
        
        
        
        scheduler.step(train_loss)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            
            # Save enhanced checkpoint with metadata
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_train_loss,
                'model_config': {
                    'input_dim': 3,
                    'edge_dim': 1,
                    'hidden_dim': hidden_dim,
                    'output_dim': 3,
                    'num_layers': num_layers,
                    'dropout': dropout
                },
                'training_config': {
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs
                }                         
            }
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with training loss: {train_loss:.6f} at '{best_model_path}'")
            
            

    print("\n--- Training Finished ---")
    print(f"Best training loss: {best_train_loss:.6f}")
    print(f"The best model is saved at: {best_model_path}")

if __name__ == '__main__':
    main()
