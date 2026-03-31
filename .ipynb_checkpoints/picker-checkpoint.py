import open3d as o3d
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QPainter, QColor

class MeshPicker(QMainWindow):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.selected_vertices = []
        
        # Set up PyQt window layout
        self.setWindowTitle("Open3D Mesh Vertex Picker")
        self.setGeometry(100, 100, 800, 600)

        # Create QWidget for embedding Open3D visualizer
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        
        self.layout = QVBoxLayout(self.widget)

        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Open3D Visualizer', width=800, height=600, visible=False)
        self.vis.add_geometry(self.mesh)

        # Add a callback function to capture mouse clicks
        self.vis.get_render_option().point_size = 10  # Increase point size for better visibility

        # Set up layout and show window
        self.widget.setLayout(self.layout)
        self.show()

        # Start the Open3D visualization in the background
        self.start_visualizer()

    def start_visualizer(self):
        """Start Open3D visualizer in background (it will be embedded into PyQt)."""
        self.vis.run()  # Run the Open3D visualizer

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # Left-click to pick a vertex
            picked_points = self.vis.get_picked_points()
            if picked_points:
                for idx in picked_points:
                    vertex = np.asarray(self.mesh.vertices)[idx]
                    if idx not in self.selected_vertices:
                        self.selected_vertices.append(idx)
                        print(f"Picked vertex {idx}: {vertex}")
                        self.highlight_selected_vertex(idx)

    def highlight_selected_vertex(self, idx):
        """Change color of the picked vertex to red."""
        vertices = np.asarray(self.mesh.vertices)
        vertices[idx] = [1, 0, 0]  # Red color for selected vertex
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.vis.update_geometry(self.mesh)

    def closeEvent(self, event):
        """Clean up when closing the window."""
        self.vis.destroy_window()
        event.accept()

def create_sample_mesh():
    """Create a sample mesh (sphere here, but you can replace with your own)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    return mesh

if __name__ == "__main__":
    # Initialize the mesh
    mesh = create_sample_mesh()

    # Initialize the MeshPicker application
    app = QApplication([])
    window = MeshPicker(mesh)
    app.exec_()
