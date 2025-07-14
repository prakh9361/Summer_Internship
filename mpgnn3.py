import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import NNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Try to import GPU-accelerated graph functions
try:
    from torch_cluster import radius_graph, knn_graph
    TORCH_CLUSTER_AVAILABLE = True
except ImportError:
    TORCH_CLUSTER_AVAILABLE = False
    print("torch_cluster not available. Using CPU fallback for graph creation.")

# ----------------- GNN Model -----------------
class AtomicDeformationNNConv(nn.Module):
    def __init__(self, input_dim=10, edge_dim=1, hidden_dim=128, output_dim=3, num_layers=4, dropout=0.1):
        super(AtomicDeformationNNConv, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.nn_input = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
        )
        self.input_conv = NNConv(input_dim, hidden_dim, self.nn_input, aggr='mean')
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        self.hidden_convs = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim * hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
            )
            self.hidden_convs.append(NNConv(hidden_dim, hidden_dim, nn_hidden, aggr='mean'))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        self.nn_output = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * output_dim, hidden_dim * output_dim)
        )
        self.output_conv = NNConv(hidden_dim, output_dim, self.nn_output, aggr='mean')

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.input_conv(x, edge_index, edge_attr)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = self.output_conv(x, edge_index, edge_attr)
        return x

# ----------------- Data Processor -----------------
class AtomicDataProcessor:
    def __init__(self, k_neighbors=1, max_distance=3.0):
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

    def load_xyz_file(self, filepath):
        try:
            data = pd.read_csv(filepath, sep='\s+', header=None).values
        except:
            data = pd.read_csv(filepath, header=None).values
        assert data.shape[0] == 4320
        return data.astype(np.float32)

    def create_graph_from_coords(self, coords):
        """
        GPU-optimized graph creation for large MD systems (4320+ atoms)
        Uses PyTorch Geometric's CUDA-accelerated functions when available
        
        Args:
            coords: array-like of shape (n_atoms, 3) with atomic coordinates
            
        Returns:
            edge_index: torch.tensor of shape (2, n_edges) with edge indices
            edge_attr: torch.tensor of shape (n_edges, 1) with edge distances
        """
        # Convert to GPU tensor if not already
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coords = coords.to(device)
        
        # Use GPU-accelerated radius graph from torch_cluster if available
        if TORCH_CLUSTER_AVAILABLE:
            try:
                edge_index = radius_graph(
                    coords, 
                    r=self.max_distance,
                    loop=False,  # No self-loops
                    max_num_neighbors=128  # Prevent memory issues with 4320 atoms
                )
                
                if edge_index.numel() == 0:  # No edges found
                    # Fallback to k-NN graph
                    edge_index = knn_graph(
                        coords, 
                        k=1,  # Just nearest neighbor
                        loop=False
                    )
                
                # Calculate edge distances on GPU
                row, col = edge_index
                edge_distances = torch.norm(coords[row] - coords[col], dim=1)
                
                # Filter by max distance if needed
                mask = edge_distances <= self.max_distance
                edge_index = edge_index[:, mask]
                edge_distances = edge_distances[mask]
                
                edge_attr = edge_distances.unsqueeze(1)
                
                return edge_index, edge_attr
                
            except Exception as e:
                print(f"GPU graph creation failed: {e}. Falling back to CPU.")
        
        # Fallback to CPU implementation
        return self._create_graph_cpu_fallback(coords.cpu().numpy())

    def _create_graph_cpu_fallback(self, coords):
        """
        CPU fallback when torch_cluster is not available or GPU fails
        Optimized version of the original function
        """
        from sklearn.neighbors import radius_neighbors_graph
        
        # Use radius graph - most efficient for large MD systems
        adj_matrix = radius_neighbors_graph(
            coords, 
            radius=self.max_distance, 
            mode='distance',
            include_self=False,
            n_jobs=-1  # Use all CPU cores
        )
        
        coo = adj_matrix.tocoo()
        
        if coo.nnz == 0:  # No edges found
            # Fallback: connect each atom to its nearest neighbor
            nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            edges = np.column_stack([np.arange(len(coords)), indices[:, 1]])
            edge_distances = distances[:, 1]
            
            # Make bidirectional
            edges_bi = np.vstack([edges, edges[:, [1, 0]]])
            edge_distances_bi = np.tile(edge_distances, 2)
        else:
            # Extract edges and distances from sparse matrix
            edges_bi = np.column_stack([coo.row, coo.col])
            edge_distances_bi = coo.data
        
        # Convert to PyTorch tensors
        edge_index = torch.from_numpy(edges_bi.T).long().contiguous()
        edge_attr = torch.from_numpy(edge_distances_bi).float().unsqueeze(1)
        
        return edge_index, edge_attr

    def create_graph_from_coords_pure_torch(self, coords):
        """
        Pure PyTorch GPU implementation - for when torch_cluster is not available
        but you want to use GPU. More memory intensive but very fast.
        """
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coords = coords.to(device)
        n_atoms = coords.shape[0]
        
        # For 4320 atoms, use chunked processing to avoid GPU memory issues
        if n_atoms > 3000:
            return self._create_graph_chunked_gpu(coords)
        
        # Compute all pairwise distances on GPU
        coords_i = coords.unsqueeze(1)  # (n_atoms, 1, 3)
        coords_j = coords.unsqueeze(0)  # (1, n_atoms, 3)
        
        # Compute distances: (n_atoms, n_atoms)
        distances = torch.norm(coords_i - coords_j, dim=2)
        
        # Create mask for valid edges (within max_distance, not self-loops)
        mask = (distances <= self.max_distance) & (distances > 0)
        
        # Get edge indices
        edge_indices = torch.nonzero(mask, as_tuple=False)  # (n_edges, 2)
        edge_distances = distances[mask]  # (n_edges,)
        
        # Handle case where no edges found
        if edge_indices.numel() == 0:
            # Connect each atom to its nearest neighbor
            distances.fill_diagonal_(float('inf'))  # Ignore self-connections
            nearest_neighbors = torch.argmin(distances, dim=1)
            
            edge_indices = torch.stack([
                torch.arange(n_atoms, device=device),
                nearest_neighbors
            ], dim=1)
            
            # Make bidirectional
            edge_indices = torch.cat([
                edge_indices,
                edge_indices.flip(1)
            ], dim=0)
            
            # Get corresponding distances
            row, col = edge_indices.t()
            edge_distances = torch.norm(coords[row] - coords[col], dim=1)
        
        edge_index = edge_indices.t().contiguous()
        edge_attr = edge_distances.unsqueeze(1)
        
        return edge_index, edge_attr

    def _create_graph_chunked_gpu(self, coords):
        """
        Chunked GPU processing for 4320 atoms to manage GPU memory
        """
        device = coords.device
        n_atoms = coords.shape[0]
        chunk_size = 1000  # Adjust based on GPU memory (lower for 4320 atoms)
        
        all_edges = []
        all_distances = []
        
        for i in range(0, n_atoms, chunk_size):
            end_i = min(i + chunk_size, n_atoms)
            chunk_coords = coords[i:end_i]
            
            # Compute distances from chunk to all atoms
            chunk_expanded = chunk_coords.unsqueeze(1)  # (chunk_size, 1, 3)
            coords_expanded = coords.unsqueeze(0)       # (1, n_atoms, 3)
            
            distances = torch.norm(chunk_expanded - coords_expanded, dim=2)
            mask = (distances <= self.max_distance) & (distances > 0)
            
            # Get indices within this chunk
            chunk_i, chunk_j = torch.nonzero(mask, as_tuple=True)
            chunk_i = chunk_i + i  # Adjust for chunk offset
            
            if len(chunk_i) > 0:
                edges_chunk = torch.stack([chunk_i, chunk_j], dim=0)
                distances_chunk = distances[mask]
                
                all_edges.append(edges_chunk)
                all_distances.append(distances_chunk)
        
        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
            edge_attr = torch.cat(all_distances, dim=0).unsqueeze(1)
        else:
            # Fallback: empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        
        return edge_index, edge_attr

    def prepare_data_pair(self, xyz_current, xyz_target, metadata_current, metadata_target, time_delta):
        coords_current = xyz_current[:, :3]
        coords_target = xyz_target[:, :3]
        metadata_current_repeated = np.tile(metadata_current, (coords_current.shape[0], 1))
        metadata_target_repeated = np.tile(metadata_target, (coords_current.shape[0], 1))
        time_delta_repeated = np.full((coords_current.shape[0], 1), time_delta, dtype=np.float32)
        node_features = np.concatenate([
            coords_current,
            metadata_current_repeated,
            metadata_target_repeated,
            time_delta_repeated
        ], axis=1)

        edge_index, edge_attr = self.create_graph_from_coords(coords_current)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(coords_target, dtype=torch.float32),
            pos=torch.tensor(coords_current, dtype=torch.float32)
        )

    def process_and_save_all(self, xyz_dir, metadata_csv, output_dir, max_frames_ahead=1):
        os.makedirs(output_dir, exist_ok=True)
        metadata_df = pd.read_csv(metadata_csv).reset_index(drop=True)
        metadata_df.columns = metadata_df.columns.str.strip()

        saved_paths = []
        for i in range(len(metadata_df)):
            current_row = metadata_df.iloc[i]
            max_target_idx = min(i + max_frames_ahead + 1, len(metadata_df))
            for j in range(i + 1, max_target_idx):
                target_row = metadata_df.iloc[j]
                time_delta = target_row['time'] - current_row['time']
                try:
                    current_xyz = self.load_xyz_file(os.path.join(xyz_dir, current_row['filename']))
                    target_xyz = self.load_xyz_file(os.path.join(xyz_dir, target_row['filename']))
                    current_meta = [current_row['volt_curr'], current_row['temp_curr'], current_row['time']]
                    target_meta = [target_row['volt_curr'], target_row['temp_curr'], target_row['time']]
                    data = self.prepare_data_pair(current_xyz, target_xyz, current_meta, target_meta, time_delta)
                    out_path = os.path.join(output_dir, f"graph_{i}_{j}.pt")
                    torch.save(data, out_path)
                    saved_paths.append(out_path)
                except Exception as e:
                    print(f"Skipping {i}->{j}: {e}")
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(metadata_df)}")
        return saved_paths

# ----------------- Training Functions -----------------
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ----------------- Main -----------------
def main():
    xyz_dir = r"D:\prakhar\model3\output_mesh_xyz\xyz_files"
    metadata_csv = r"D:\prakhar\model3\voltage_temperature_data.csv"
    graph_data_dir = r"D:\prakhar\model3\graph_data"
    epochs = 10
    batch_size = 1
    hidden_dim = 64
    num_layers = 10
    lr = 0.001
    save_model = "atomic_nnconv_model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if TORCH_CLUSTER_AVAILABLE:
        print("GPU-accelerated graph creation available (torch_cluster)")
    else:
        print("Using CPU fallback for graph creation")

    processor = AtomicDataProcessor()
    if not os.path.exists(graph_data_dir) or not os.listdir(graph_data_dir):
        print("Preprocessing data...")
        processor.process_and_save_all(xyz_dir, metadata_csv, graph_data_dir)

    data_files = [os.path.join(graph_data_dir, f) for f in os.listdir(graph_data_dir) if f.endswith(".pt")]
    data_list = [torch.load(f) for f in data_files]
    print(f"Loaded {len(data_list)} preprocessed graph pairs.")

    random.shuffle(data_list)
    train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = AtomicDeformationNNConv(
        input_dim=10,
        edge_dim=1,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Train Loss: {train_loss:.6f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), save_model)
            print(f"Model saved at epoch {epoch+1}")

if __name__ == "__main__":
    main()