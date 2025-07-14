#!/usr/bin/env python3
"""
XYZ to Alpha Shape Converter
Reads XYZ point cloud files and generates alpha shape meshes
"""

import numpy as np
import open3d as o3d
import argparse
import os
import sys

def read_xyz_file(filename):
    """Read XYZ file and return points as numpy array"""
    try:
        # Try reading as space-separated values
        points = np.loadtxt(filename)
        
        # Handle different XYZ formats
        if points.shape[1] < 3:
            raise ValueError(f"XYZ file must have at least 3 columns (x, y, z), got {points.shape[1]}")
        
        # Take only first 3 columns (x, y, z) in case there are additional columns
        points = points[:, :3]
        
        print(f"Loaded {len(points)} points from {filename}")
        return points
        
    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        sys.exit(1)

def create_alpha_shape_mesh(points, alpha=None, verbose=True):
    """Create alpha shape with automatic parameter tuning"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calculate average nearest neighbor distance to estimate good alpha
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    if alpha is None:
        # Auto-determine alpha based on point cloud characteristics
        alpha_values = [avg_dist, avg_dist * 2, avg_dist * 4, avg_dist * 8]
    else:
        alpha_values = [alpha]
    
    if verbose:
        print(f"Average nearest neighbor distance: {avg_dist:.4f}")
        print(f"Trying alpha values: {[f'{a:.4f}' for a in alpha_values]}")
    
    # Estimate normals for better mesh quality
    pcd.estimate_normals()
    
    # Try different alpha values until we get a valid mesh
    for alpha_val in alpha_values:
        try:
            if verbose:
                print(f"Trying alpha = {alpha_val:.4f}")
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_val)
            
            if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                if verbose:
                    print(f"Success with alpha = {alpha_val:.4f}")
                    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                return mesh, alpha_val
            else:
                if verbose:
                    print(f"Empty mesh with alpha = {alpha_val:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"Failed with alpha = {alpha_val:.4f}: {e}")
            continue
    
    # If all alpha values fail, fall back to Poisson reconstruction
    if verbose:
        print("All alpha values failed, trying Poisson reconstruction")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)
        
        # Remove low-density vertices
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            if verbose:
                print(f"Poisson reconstruction successful: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh, "poisson"
        
    except Exception as e:
        if verbose:
            print(f"Poisson reconstruction failed: {e}")
    
    # Ultimate fallback - return convex hull
    if verbose:
        print("Using convex hull as final fallback")
    try:
        mesh, _ = pcd.compute_convex_hull()
        if verbose:
            print(f"Convex hull: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh, "convex_hull"
    except Exception as e:
        print(f"All mesh generation methods failed: {e}")
        return None, None

def save_mesh(mesh, output_path, format_type="obj"):
    """Save mesh to file"""
    format_type.lower() == "obj"
    success = o3d.io.write_triangle_mesh(output_path, mesh)
        
    print(f"Mesh saved to: {output_path}")

def visualize_results(points, mesh):
    """Visualize original points and generated mesh"""
    # Create point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])  # Red points
    
    # Color the mesh
    mesh.paint_uniform_color([0, 0.7, 0.7])  # Cyan mesh
    mesh.compute_vertex_normals()
    
    print("Visualizing results... Close the window to continue.")
    o3d.visualization.draw_geometries([pcd, mesh], 
                                    window_name="Alpha Shape Result",
                                    width=800, height=600)

def main():
    parser = argparse.ArgumentParser(description="Convert XYZ point cloud to alpha shape mesh")
    parser.add_argument("-o", "--output", help="Output mesh file path (default: input_name_alpha_shape.obj)")
    parser.add_argument("-a", "--alpha", type=float, help="Alpha parameter (auto-determined if not specified)")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the result")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    # Read XYZ file
    points = read_xyz_file("predicted_output.xyz")
    
    # Generate output filename if not provided
    if args.output is None:
        args.output = f"infered_alpha_shape.obj"
    
    # Create alpha shape
    verbose = not args.no_verbose
    mesh, alpha_used = create_alpha_shape_mesh(points, args.alpha, verbose)
    
    if mesh is None:
        print("Failed to generate alpha shape")
        sys.exit(1)
    
    # Save mesh
    save_mesh(mesh, args.output)
    
    # Visualize if requested
    if args.visualize:
        visualize_results(points, mesh)
    
    if verbose:
        print(f"\nSummary:")
        print(f"Input points: {len(points)}")
        print(f"Output vertices: {len(mesh.vertices)}")
        print(f"Output triangles: {len(mesh.triangles)}")
        print(f"Alpha parameter used: {alpha_used}")
        print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
