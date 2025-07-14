import torch
import numpy as np
from torch_geometric.loader import DataLoader
from mpgnn4 import AtomicDataProcessor
from mpgnn4 import AtomicDeformationNNConv

def run_inference(xyz_file_path, volt_current, temp_current, time_current, volt_target, temp_target, time_target):
    # Model parameters (must match training)
    hidden_dim = 64
    num_layers = 3
    model_path = "optimized_atomic_gnn_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = AtomicDeformationNNConv(
        input_dim=10,
        edge_dim=1,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_layers=num_layers
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Process input data
    processor = AtomicDataProcessor()
    
    # Load XYZ file
    xyz_data = processor.load_xyz_file(xyz_file_path)
    coords = xyz_data[:, :3]  # Extract coordinates
    
    # Create metadata arrays (matching training format)
    current_meta = [volt_current, temp_current, time_current]
    target_meta = [volt_target, temp_target, time_target]
    time_delta = time_target - time_current
    
    # Create node features (same as training)
    metadata_current_repeated = np.tile(current_meta, (coords.shape[0], 1))
    metadata_target_repeated = np.tile(target_meta, (coords.shape[0], 1))
    time_delta_repeated = np.full((coords.shape[0], 1), time_delta, dtype=np.float32)
    
    node_features = np.concatenate([
        coords,                      # [4320, 3] - current coordinates
        metadata_current_repeated,   # [4320, 3] - current conditions
        metadata_target_repeated,    # [4320, 3] - target conditions  
        time_delta_repeated         # [4320, 1] - time difference
    ], axis=1)  # Final shape: [4320, 10]
    
    # Create graph edges
    edge_index, edge_attr = processor.create_graph_from_coords(coords)
    
    # Create data object
    from torch_geometric.data import Data
    graph_data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Run inference
    with torch.no_grad():
        graph_data = graph_data.to(device)
        predicted_coords = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        
        print(f"Input conditions:")
        print(f"  Current: {volt_current}V, {temp_current}K, t={time_current}")
        print(f"  Target: {volt_target}V, {temp_target}K, t={time_target}")
        print(f"  Time delta: {time_delta}")
        print(f"Predicted coordinates shape: {predicted_coords.shape}")
        print(f"Sample predicted positions (first 3 atoms):")
        print(predicted_coords[:3].cpu().numpy())
        
        return predicted_coords.cpu()

def save_coords_as_xyz(coords, output_path, comment="Predicted deformed structure"):
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    
    with open(output_path, 'w', encoding='utf-8') as f:  # <--- add encoding='utf-8'
        for i, (x, y, z) in enumerate(coords):
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"XYZ file saved to: {output_path}")


def run_inference_and_save(xyz_file_path, volt_current, temp_current, time_current, 
                          volt_target, temp_target, time_target, output_xyz_path):
    """
    Run inference and save result as XYZ file
    """
    predicted_coords = run_inference(xyz_file_path, volt_current, temp_current, time_current,
                                   volt_target, temp_target, time_target)
    
    time_delta = time_target - time_current
    comment = f"Deformed structure: {volt_current}V→{volt_target}V, {temp_current}K→{temp_target}K, {time_current}→{time_target} (dt={time_delta})"
    save_coords_as_xyz(predicted_coords, output_xyz_path, comment)
    
    return predicted_coords

def predict_deformation_sequence_and_save(xyz_file_path, voltage_sequence, temperature_sequence, 
                                        time_sequence, output_dir="deformation_sequence"):
    """
    Predict a sequence of deformations and save each as XYZ file
    
    Args:
        xyz_file_path: Initial XYZ structure
        voltage_sequence: List of voltage values
        temperature_sequence: List of temperature values
        time_sequence: List of time values
        output_dir: Output directory for XYZ files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = []
    current_xyz = xyz_file_path
    
    for i in range(len(voltage_sequence) - 1):
        volt_curr = voltage_sequence[i]
        temp_curr = temperature_sequence[i]
        time_curr = time_sequence[i]
        volt_target = voltage_sequence[i + 1]
        temp_target = temperature_sequence[i + 1]
        time_target = time_sequence[i + 1]
        
        output_path = os.path.join(output_dir, f"deformed_step_{i+1:03d}.xyz")
        pred = run_inference_and_save(current_xyz, volt_curr, temp_curr, time_curr,
                                    volt_target, temp_target, time_target, output_path)
        predictions.append(pred)
        
        time_delta = time_target - time_curr
        print(f"Step {i+1}: {volt_curr}V→{volt_target}V, {temp_curr}K→{temp_target}K, t={time_curr}→{time_target} (Δt={time_delta})")
    
    print(f"All deformation steps saved in: {output_dir}")
    return predictions

# Example usage functions
def simple_voltage_sweep_and_save(xyz_file, start_voltage, end_voltage, temperature, start_time, end_time, num_steps=5, output_dir="voltage_sweep"):
    """Sweep voltage while keeping temperature constant and save XYZ files"""
    voltages = np.linspace(start_voltage, end_voltage, num_steps)
    temperatures = [temperature] * num_steps
    times = np.linspace(start_time, end_time, num_steps)
    
    return predict_deformation_sequence_and_save(xyz_file, voltages, temperatures, times, output_dir)

def simple_temperature_sweep_and_save(xyz_file, voltage, start_temp, end_temp, start_time, end_time, num_steps=5, output_dir="temp_sweep"):
    """Sweep temperature while keeping voltage constant and save XYZ files"""
    voltages = [voltage] * num_steps
    temperatures = np.linspace(start_temp, end_temp, num_steps)
    times = np.linspace(start_time, end_time, num_steps)
    
    return predict_deformation_sequence_and_save(xyz_file, voltages, temperatures, times, output_dir)

if __name__ == "__main__":
    # Example usage
    xyz_file = r"D:\prakhar\model2\output_mesh_xyz\xyz_files\frame_001059.xyz"  
    
    # Single prediction and save as XYZ

    prediction = run_inference_and_save(
        xyz_file_path=xyz_file,
        volt_current=37.96,     # Starting voltage
        temp_current=1076.34,   # Starting temperature
        time_current=529.5,     # Starting time
        volt_target=38.69,      # Target voltage
        temp_target=1100.0,    # Target temperature
        time_target=530.0,      # Target time
        output_xyz_path="deformed_output.xyz"
    )
    
    # Voltage sweep example - saves multiple XYZ files
    voltage_sweep_results = simple_voltage_sweep_and_save(
        xyz_file, start_voltage=0.0, end_voltage=10.0, temperature=300.0, 
        start_time=0.0, end_time=5.0, num_steps=6, output_dir="voltage_sweep_results"
    )
    
    # Temperature sweep example - saves multiple XYZ files
    temp_sweep_results = simple_temperature_sweep_and_save(
        xyz_file, voltage=5.0, start_temp=300.0, end_temp=500.0,
        start_time=0.0, end_time=5.0, num_steps=6, output_dir="temp_sweep_results"
    )