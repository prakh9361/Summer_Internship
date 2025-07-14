import open3d as o3d
import numpy as np
import os

class LAMMPSTrajectoryParser:
    """Simple LAMMPS trajectory parser"""
    
    def __init__(self, filename):
        self.filename = filename
        self.frames = []
        self._parse_trajectory()
    
    def _parse_trajectory(self):
        """Parse the LAMMPS trajectory file"""
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if "ITEM: TIMESTEP" in lines[i]:
                frame_data = {}
                i += 1
                frame_data['timestep'] = int(lines[i].strip())
                i += 1
                
                # Get number of atoms
                if "ITEM: NUMBER OF ATOMS" in lines[i]:
                    i += 1
                    frame_data['natoms'] = int(lines[i].strip())
                    i += 1
                
                # Skip box bounds
                if "ITEM: BOX BOUNDS" in lines[i]:
                    i += 4  # Skip box bounds lines
                
                # Get atom data
                if "ITEM: ATOMS" in lines[i]:
                    i += 1
                    atoms = []
                    for j in range(frame_data['natoms']):
                        parts = lines[i + j].strip().split()
                        # Assuming format: id type x y z
                        atoms.append({
                            'id': int(parts[0]),
                            'type': int(parts[1]),
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'z': float(parts[4])
                        })
                    frame_data['atoms'] = atoms
                    i += frame_data['natoms']
                
                self.frames.append(frame_data)
            else:
                i += 1
    
    def get_num_frames(self):
        return len(self.frames)
    
    def get_atom_positions(self, frame_idx, dimensions=3, atom_types=None):
        """Get atom positions for a specific frame, optionally filtered by atom type"""
        if frame_idx >= len(self.frames):
            return {}
        
        frame = self.frames[frame_idx]
        positions = {}
        
        for atom in frame['atoms']:
            # Filter by atom type if specified
            if atom_types is not None and atom['type'] not in atom_types:
                continue
                
            atom_id = atom['id']
            if dimensions == 3:
                positions[atom_id] = [atom['x'], atom['y'], atom['z']]
            elif dimensions == 2:
                positions[atom_id] = [atom['x'], atom['y']]
        
        return positions

class LAMMPSToXYZConverter:
    """Simple converter that outputs only XYZ files"""
    
    def __init__(self, trj_file, output_dir="xyz_output", atom_types=None):
        self.trj_file = trj_file
        self.output_dir = output_dir
        self.atom_types = atom_types  # List of atom types to include (e.g., [1, 2])
        self.parser = LAMMPSTrajectoryParser(trj_file)
        
        os.makedirs(output_dir, exist_ok=True)
        
    def process_all_frames(self, start_frame=0, end_frame=None, step=1):
        """Process all frames and save as XYZ files"""
        
        total_frames = self.parser.get_num_frames()
        if end_frame is None:
            end_frame = total_frames
        
        successful_frames = 0
        
        for frame_idx in range(start_frame, min(end_frame, total_frames), step):
            try:
                xyz_file = os.path.join(self.output_dir, f"frame_{frame_idx:06d}.xyz")
                
                if self.process_single_frame(frame_idx, xyz_file):
                    successful_frames += 1
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
        
        return successful_frames

    def process_single_frame(self, frame_index, output_xyz_file):
        """Process a single frame and save as XYZ file"""
        
        positions_dict = self.parser.get_atom_positions(frame_index, 3, self.atom_types)
        if not positions_dict:
            return False
        
        points = np.array(list(positions_dict.values()), dtype=np.float64)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Save XYZ file
        o3d.io.write_point_cloud(output_xyz_file, pcd, write_ascii=True)
        
        return True

def convert_trajectory_to_xyz(trj_file, output_dir="xyz_output", atom_types=None, start_frame=0, end_frame=None, step=1):
    """Convert LAMMPS trajectory to XYZ files
    
    Parameters:
    - trj_file: input LAMMPS trajectory file
    - output_dir: output directory for XYZ files
    - atom_types: list of atom types to include (e.g., [1, 2] or [1] for only type 1)
    - start_frame, end_frame, step: frame range parameters
    """
    
    converter = LAMMPSToXYZConverter(trj_file, output_dir, atom_types)
    successful_frames = converter.process_all_frames(start_frame=start_frame, end_frame=end_frame, step=step)
    
    return successful_frames

if __name__ == "__main__":
    trj_file = "voltage_cycle4_temp_16.208.lammpstrj"
    output_directory = "xyz_output"
    
    # Specify which atom types to include (e.g., [1] for only type 1, [1,2] for types 1 and 2)
    # Set to None to include all atom types
    target_atom_types = [3]  # Change this to your desired atom types
    
    try:
        successful_count = convert_trajectory_to_xyz(
            trj_file=trj_file,
            output_dir=output_directory,
            atom_types=target_atom_types,
            step=1 
        )
        
        if target_atom_types:
            print(f"Successfully converted {successful_count} frames to XYZ format (atom types: {target_atom_types})")
        else:
            print(f"Successfully converted {successful_count} frames to XYZ format (all atom types)")
        print(f"Output directory: {output_directory}")
        
    except Exception as e:
        print(f"Error: {e}")