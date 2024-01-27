import open3d as o3d
import os

# Create a simple 3D cube
mesh = o3d.geometry.TriangleMesh.create_box()

# Create a visualizer object
vis = o3d.visualization.Visualizer()

# Add the cube mesh to the visualizer
vis.create_window()
vis.add_geometry(mesh)

# Run the visualization (opens a window to display the cube)
vis.run()

# Close the visualizer window
vis.destroy_window()