import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm

class XYZDataset(Dataset):
    """Dataset class for loading and preprocessing XYZ files"""
    
    def __init__(self, data_dir, sequence_length=100, transform=None):
        """
        Args:
            data_dir: Directory containing XYZ files
            sequence_length: Number of frames to predict ahead
            transform: Optional transform to apply to data
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Get all XYZ files and sort them
        self.xyz_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.xyz")))
        
        # Load and preprocess all data
        self.data = self._load_all_data()
        self.scaler = StandardScaler()
        self.data_normalized = self._normalize_data()
        
        # Create input-target pairs
        self.pairs = self._create_pairs()
        
    def _load_xyz_file(self, filepath):
        """Load a single XYZ file and return coordinates"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            coords = []
            
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 3:  # Ensure we have at least x, y, z
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coords.extend([x, y, z])
            
            return np.array(coords)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _load_all_data(self):
        """Load all XYZ files into memory"""
        print("Loading XYZ files...")
        data = []
        
        for filepath in tqdm(self.xyz_files):
            coords = self._load_xyz_file(filepath)
            if coords is not None:
                data.append(coords)
        
        return np.array(data)
    
    def _normalize_data(self):
        """Normalize the data using StandardScaler"""
        print("Normalizing data...")
        # Flatten for fitting scaler
        data_flat = self.data.reshape(-1, self.data.shape[-1])
        data_normalized = self.scaler.fit_transform(data_flat)
        return data_normalized.reshape(self.data.shape)
    
    def _create_pairs(self):
        """Create input-target pairs for training"""
        pairs = []
        
        # Create pairs where input is frame i and target is frame i+sequence_length
        for i in range(len(self.data_normalized) - self.sequence_length):
            input_frame = self.data_normalized[i]
            target_frame = self.data_normalized[i + self.sequence_length]
            pairs.append((input_frame, target_frame))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_frame, target_frame = self.pairs[idx]
        
        if self.transform:
            input_frame = self.transform(input_frame)
            target_frame = self.transform(target_frame)
        
        return torch.FloatTensor(input_frame), torch.FloatTensor(target_frame)

class FramePredictionCNN(nn.Module):
    """CNN model for predicting future frames"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.0):
        """
        Args:
            input_dim: Dimension of input frame (number of coordinates)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(FramePredictionCNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer
        layers.extend([
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        # Build decoder layers
        decoder_dims = hidden_dims[::-1]  # Reverse the hidden dims
        prev_dim = prev_dim // 2
        
        for hidden_dim in decoder_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CollapsePreventionLoss(nn.Module):
    """Loss function to prevent atoms from collapsing together"""
    
    def __init__(self, min_distance=2.9, loss_weight=1.0):
        """
        Args:
            min_distance: Minimum allowed distance between atoms
            loss_weight: Weight for this loss component
        """
        super(CollapsePreventionLoss, self).__init__()
        self.min_distance = min_distance
        self.loss_weight = loss_weight
    
    def forward(self, coordinates):
        """
        Calculate collapse prevention loss
        
        Args:
            coordinates: Tensor of shape (batch_size, num_coords) where num_coords = num_atoms * 3
        """
        batch_size = coordinates.shape[0]
        num_atoms = coordinates.shape[1] // 3
        
        # Reshape to (batch_size, num_atoms, 3)
        coords_reshaped = coordinates.view(batch_size, num_atoms, 3)
        
        total_loss = 0.0
        
        for b in range(batch_size):
            atoms = coords_reshaped[b]  # (num_atoms, 3)
            
            # Calculate pairwise distances
            # Expand dimensions for broadcasting
            atoms_i = atoms.unsqueeze(1)  # (num_atoms, 1, 3)
            atoms_j = atoms.unsqueeze(0)  # (1, num_atoms, 3)
            
            # Calculate squared distances
            dist_squared = torch.sum((atoms_i - atoms_j) ** 2, dim=2)  # (num_atoms, num_atoms)
            
            # Get upper triangular part (avoid double counting and self-distances)
            mask = torch.triu(torch.ones_like(dist_squared), diagonal=1).bool()
            pairwise_dist_squared = dist_squared[mask]
            
            # Calculate distances
            pairwise_distances = torch.sqrt(pairwise_dist_squared + 1e-8)  # Add small epsilon for stability
            
            # Penalize distances smaller than min_distance
            violations = torch.clamp(self.min_distance - pairwise_distances, min=0.0)
            
            # Use squared violations for stronger penalty
            batch_loss = torch.sum(violations ** 2)
            total_loss += batch_loss
        
        return self.loss_weight * total_loss / batch_size

class CombinedLoss(nn.Module):
    """Combined loss function with MSE and collapse prevention"""
    
    def __init__(self, mse_weight=1.0, collapse_weight=0.1, min_distance=2.9):
        """
        Args:
            mse_weight: Weight for MSE loss
            collapse_weight: Weight for collapse prevention loss
            min_distance: Minimum allowed distance between atoms
        """
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.collapse_loss = CollapsePreventionLoss(min_distance, collapse_weight)
        self.mse_weight = mse_weight
        self.collapse_weight = collapse_weight
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss
        
        Args:
            predictions: Predicted coordinates
            targets: Target coordinates
        """
        # MSE loss between predictions and targets
        mse = self.mse_loss(predictions, targets)
        
        # Collapse prevention loss on predictions
        collapse_pred = self.collapse_loss(predictions)
        
        # Optional: Also apply collapse loss to targets (for consistency)
        collapse_target = self.collapse_loss(targets)
        
        total_loss = (self.mse_weight * mse + 
                     collapse_pred + 
                     0.1 * collapse_target)  # Small weight for target consistency
        
        return total_loss, mse, collapse_pred, collapse_target

class FramePredictor:
    """Main class for training and using the frame prediction model"""
    
    def __init__(self, data_dir, sequence_length=100, batch_size=32, learning_rate=0.001, 
                 collapse_weight=0.1, min_distance=2.9):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.collapse_weight = collapse_weight
        self.min_distance = min_distance
        
        # Initialize dataset
        self.dataset = XYZDataset(data_dir, sequence_length)
        
        # Split dataset
        train_size = int(1 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Initialize model
        input_dim = self.dataset.data.shape[1]  # Number of coordinates per frame
        self.model = FramePredictionCNN(input_dim)
        
        # Combined loss function with collapse prevention
        self.criterion = CombinedLoss(
            mse_weight=1.0, 
            collapse_weight=collapse_weight, 
            min_distance=min_distance
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.mse_losses = []
        self.collapse_losses = []
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        print(f"Model initialized with input dimension: {input_dim}")
        print(f"Training on device: {self.device}")
        print(f"Dataset size: {len(self.dataset)} pairs")
        print(f"Train/Val split: {len(self.train_dataset)}/{max(1,len(self.val_dataset))}")
        print(f"Collapse prevention: min_distance={min_distance}, weight={collapse_weight}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_collapse = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate combined loss
            loss, mse_loss, collapse_pred, collapse_target = self.criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_collapse += collapse_pred.item()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mse = total_mse / len(self.train_loader)
        avg_collapse = total_collapse / len(self.train_loader)
        
        return avg_loss, avg_mse, avg_collapse
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_collapse = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                # Calculate combined loss
                loss, mse_loss, collapse_pred, collapse_target = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_collapse += collapse_pred.item()
        
        avg_loss = total_loss 
        avg_mse = total_mse 
        avg_collapse = total_collapse 
        
        return avg_loss, avg_mse, avg_collapse
    
    def train(self, num_epochs=100, save_path="frame_predictor.pth"):
        """Train the model"""
        print(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(num_epochs):
            train_loss,_,_ = self.train_epoch()
            val_loss,_,_ = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(train_loss)
            
            # Early stopping
            """if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler': self.dataset.scaler,
                    'input_dim': self.dataset.data.shape[1],
                    'sequence_length': self.sequence_length
                }, save_path)
            else:
                patience_counter += 1"""
            
            """if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break"""
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        print("Training completed!")
        return self.train_losses, self.val_losses
    
    def predict(self, input_frame):
        """Predict future frame given current frame"""
        self.model.eval()
        
        # Normalize input if it's raw coordinates
        if isinstance(input_frame, np.ndarray):
            input_normalized = self.dataset.scaler.transform(input_frame.reshape(1, -1))
            input_tensor = torch.FloatTensor(input_normalized).to(self.device)
        else:
            input_tensor = input_frame.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
            # Denormalize prediction
            prediction_np = prediction.cpu().numpy()
            prediction_denorm = self.dataset.scaler.inverse_transform(prediction_np)
        
        return prediction_denorm
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()

def load_trained_model(model_path, device=None):
    """Load a trained model for inference"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = FramePredictionCNN(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint

def run_inference_on_trained_model(model_path, input_xyz_file=None, input_frame_idx=None, data_dir=None):
    """
    Run inference on a trained model
    
    Args:
        model_path: Path to saved model (.pth file)
        input_xyz_file: Path to specific XYZ file to use as input (optional)
        input_frame_idx: Index of frame from training data to use as input (optional)
        data_dir: Original data directory (needed if using frame index)
    """
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model architecture
    model = FramePredictionCNN(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    scaler = checkpoint['scaler']
    sequence_length = checkpoint['sequence_length']
    
    print(f"Loaded model predicting {sequence_length} frames ahead")
    print(f"Input dimension: {checkpoint['input_dim']}")
    
    # Get input frame
    if input_xyz_file:
        # Load from specific XYZ file
        print(f"Loading input from: {input_xyz_file}")
        input_coords = load_single_xyz_file(input_xyz_file)
    elif input_frame_idx is not None and data_dir:
        # Load from training data
        dataset = XYZDataset(data_dir, sequence_length)
        input_coords = dataset.data[input_frame_idx]
        print(f"Using frame {input_frame_idx} from training data")
    else:
        raise ValueError("Must provide either input_xyz_file or (input_frame_idx + data_dir)")
    
    # Normalize input
    input_normalized = scaler.transform(input_coords.reshape(1, -1))
    input_tensor = torch.FloatTensor(input_normalized).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction_normalized = model(input_tensor)
        prediction = scaler.inverse_transform(prediction_normalized.cpu().numpy())
    
    return input_coords, prediction.squeeze(), model, scaler

def load_single_xyz_file(filepath):
    """Load coordinates from a single XYZ file"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        coords = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    coords.extend([x, y, z])
        
        return np.array(coords)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_prediction_as_xyz(coords, output_path):
    """Save coordinates as XYZ file"""
    # Reshape coordinates to (N, 3) format
    coords_reshaped = coords.reshape(-1, 3)
    
    with open(output_path, 'w') as f:
        for x, y, z in coords_reshaped:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Prediction saved to: {output_path}")

def analyze_distances(coords, min_distance=2.9):
    """Analyze pairwise distances in coordinates"""
    # Reshape to (N, 3) format
    coords_reshaped = coords.reshape(-1, 3)
    num_atoms = coords_reshaped.shape[0]
    
    distances = []
    violations = 0
    min_dist = float('inf')
    
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            dist = np.linalg.norm(coords_reshaped[i] - coords_reshaped[j])
            distances.append(dist)
            if dist < min_distance:
                violations += 1
            min_dist = min(min_dist, dist)
    
    
    """Visualize input and predicted coordinates (works for 2D/3D data)"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Reshape to (N, 3) format
        input_reshaped = input_coords.reshape(-1, 3)
        pred_reshaped = predicted_coords.reshape(-1, 3)
        
        fig = plt.figure(figsize=(15, 5))
        
        # Plot input frame
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(input_reshaped[:, 0], input_reshaped[:, 1], input_reshaped[:, 2], 
                   c='blue', alpha=0.6, s=50)
        ax1.set_title('Input Frame')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot predicted frame
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pred_reshaped[:, 0], pred_reshaped[:, 1], pred_reshaped[:, 2], 
                   c='red', alpha=0.6, s=50)
        ax2.set_title('Predicted Frame')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Plot both together
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(input_reshaped[:, 0], input_reshaped[:, 1], input_reshaped[:, 2], 
                   c='blue', alpha=0.6, s=50, label='Input')
        ax3.scatter(pred_reshaped[:, 0], pred_reshaped[:, 1], pred_reshaped[:, 2], 
                   c='red', alpha=0.6, s=50, label='Predicted')
        ax3.set_title('Input vs Predicted')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

    return {
        'distances': distances,
        'violations': violations,
        'min_distance': min_dist,
        'mean_distance': np.mean(distances),
        'total_pairs': len(distances)
    }

# Example usage
if __name__ == "__main__":
    
    # TRAINING PHASE
    print("="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    # Set your data directory path
    DATA_DIR = "xyz_files"
    
    # Initialize and train the model with collapse prevention
    predictor = FramePredictor(
        data_dir=DATA_DIR,
        sequence_length=100,  # Predict 100 frames ahead
        batch_size=32,
        learning_rate=0.001,
        collapse_weight=0.9,  # Weight for collapse prevention loss
        min_distance=2.9      # Minimum allowed distance between atoms
    )
    
    # Train the model
    train_losses, val_losses = predictor.train(num_epochs=1000)
   
    
    # Plot training history
    predictor.plot_training_history()
    
    model_save_path = "trained_model.pth"
    predictor.save_model(model_save_path)
    print(f"\nModel saved to {model_save_path}")
    print("Training completed and model saved!")
    
    # INFERENCE PHASE
    print("="*50)
    print("INFERENCE PHASE")
    print("="*50)
    
    # Method 1: Use a frame from the training data
    print("\n1. Inference using training data frame:")
    input_coords, prediction, model, scaler = run_inference_on_trained_model(
        model_path="frame_predictor.pth",
        input_frame_idx=0,  # Use first frame
        data_dir=DATA_DIR
    )
    
    print(f"Input shape: {input_coords.shape}")
    print(f"Prediction shape: {prediction.shape}")
    
    # Save prediction
    save_prediction_as_xyz(prediction, "predicted_frame.xyz")
    
    # Visualize
    visualize_prediction(input_coords, prediction, "prediction_visualization.png")
    
   
    
    
    print("\n3. Batch inference on multiple frames:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("frame_predictor.pth", map_location=device)
    model = FramePredictionCNN(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load dataset for batch inference
    dataset = XYZDataset(DATA_DIR, checkpoint['sequence_length'])
    
    # Predict on first 5 frames
    for i in range(5):
        input_frame = dataset.data[i]
        input_normalized = dataset.scaler.transform(input_frame.reshape(1, -1))
        input_tensor = torch.FloatTensor(input_normalized).to(device)
        
        with torch.no_grad():
            pred_normalized = model(input_tensor)
            prediction = dataset.scaler.inverse_transform(pred_normalized.cpu().numpy())
        
        # Save each prediction
        save_prediction_as_xyz(prediction.squeeze(), f"batch_prediction_{i:03d}.xyz")
        print(f"Processed frame {i}")
    
    print("\nInference completed!")