import open3d as o3d
import cv2

#path = '/home/vincent/PycharmProjects/bundlesdf_utility/data/textured_mesh.obj'


# path = 'C://Users//Vincent//Downloads//final_reconstruction//textured_mesh.obj'
# path = 'C://Users//Vincent//Downloads//milk//milk.obj'
# path = 'C:Users//Vincent//Downloads//data//data//beautiful_1.obj'

path = 'C://Users//Vincent//Downloads//capsule//capsule.obj'


# img_path = 'C://Users//Vincent//Downloads//capsule//capsule.jpg'

# mesh = o3d.io.read_triangle_mesh(path, True)
# img = cv2.imread(img_path)
# mesh.textures = [o3d.cpu.pybind.geometry.Image(img)]
# mesh.compute_vertex_normals()
#
# drawer = o3d.visualization.draw(mesh)

mesh = o3d.io.read_triangle_mesh("capsule.obj", True)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])