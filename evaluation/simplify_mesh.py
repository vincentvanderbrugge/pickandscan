import open3d as o3d
import argparse

if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-f', '--factor', type=float)
    args = parser.parse_args()

    # Load
    mesh_in = o3d.io.read_triangle_mesh(args.input)
    num_vertices_input = len(mesh_in.vertices)

    # Simplify
    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / args.factor
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    mesh_smp.compute_vertex_normals()

    # Visualize output
    o3d.visualization.draw_geometries([mesh_smp])

    # Save output
    o3d.io.write_triangle_mesh(args.output, mesh_smp)

    # Print result
    num_vertices_output = len(mesh_smp.vertices)
    print(f"Simplified from {num_vertices_input} to {num_vertices_output} vertices.")
