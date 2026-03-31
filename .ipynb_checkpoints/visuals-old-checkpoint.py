import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import open3d as o3d
import torch
from differential import *

import colourmappings
import importlib
importlib.reload(colourmappings)
from colourmappings import *

from bns_utils import *
from mesh_processing import *


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


import implicit_reps
import importlib
importlib.reload(implicit_reps)
from implicit_reps import *
        
from mpl_toolkits.mplot3d import Axes3D



def write_custom_colour_ply_file(tm=None, colouringdict=None, filepath=None):
	
	from plyfile import PlyData, PlyElement
	tm.export('rendering/rendering_results/temp.ply')
	p = PlyData.read('rendering/rendering_results/temp.ply')
	v = p.elements[0]
	f = p.elements[1]
	
	# Create the new vertex data with appropriate dtype
	extra_data_specs = [ (colouringname, 'f8', (3,)) for colouringname in colouringdict.keys() ]
	a = np.empty(len(v.data), v.data.dtype.descr + extra_data_specs)
	for name in v.data.dtype.fields:
	    a[name] = v[name]
	
	for colouringname,colouring in colouringdict.items():    
		a[colouringname] = colouring[:,:3]
	
	# Recreate the PlyElement instance
	v = PlyElement.describe(a, 'vertex')
	
	# Recreate the PlyData instance
	p = PlyData([v, f], text=True)
	p.write(filepath)




def show_f_list(f_list):
    #Input: a list of pointcloud dictionaries, possibly with colours.
    #E.g. show_f_list([{'points':np.random.rand(100,1000,3), 'colors':np.random.rand(100,1000,3)}])
    
    pcds = []
    for f in f_list:
        num_triangles = f['points'].shape[0]
        
        # Flatten points and colors to create a single point cloud
        points = f['points'].reshape(-1, 3)  # Shape: e.g. [(80 * 4330), 3]
        
        
        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = Vector3dVector(points)   # Assign points
        if not (f['colors'] is None):
            colors = f['colors'].reshape(-1,3)
            pcd.colors = Vector3dVector(colors)   # Assign per-point colors
            
        pcds.append(pcd)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries(pcds, window_name="Colored 3D Point Cloud")





def save_mesh_screenshots(meshes, output_prefix="screenshot", width=3200, height=2400,
                          background_color=[1,1,1], keep_window_open=True ):
        """
        Save screenshots of the given meshes from X, Y, and Z axis views.
        
        Parameters
        ----------
        meshes : list of o3d.geometry.TriangleMesh
            The meshes to display.
        output_prefix : str
            Prefix for screenshot filenames; '_x.png', '_y.png', '_z.png' will be appended.
        width : int
            Window width in pixels.
        height : int
            Window height in pixels.
        background_color : list of float
            RGB background color, each in [0,1].
        keep_window_open : bool
            If True, keeps the Open3D window open at the end.
        use_custom_light : bool
            If True, uses a custom light direction instead of the default camera-based lighting.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Open3D Screenshot", width=width, height=height)
        
        

        for mesh in meshes:
            mesh.compute_vertex_normals()
            mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(mesh.vertex_colors), 0, 1) )
            
        for mesh in meshes:
            vis.add_geometry(mesh)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = False
        opt.point_size = 0.0
        opt.background_color = np.asarray([0.8, 0.8, 0.8])
        
        
        opt.light_on = True
        
        ctr = vis.get_view_control()
        
        # Compute center of all meshes for lookat
        all_points = np.vstack([np.asarray(mesh.vertices) for mesh in meshes])
        center = all_points.mean(axis=0)
        
        # Define camera views: front, side, top
        camera_views = {
            'x': {'front': [1, 0, 0], 'up': [0, 0, 1]},   # looking along +X
            'y': {'front': [0, 1, 0], 'up': [0, 0, 1]},   # looking along +Y
            'z': {'front': [0, 0, 1], 'up': [0, 1, 0]},   # looking along +Z
        }
        
        
        
        for axis, params in camera_views.items():
            ctr.set_lookat(center)
            ctr.set_front(params['front'])
            ctr.set_up(params['up'])
            ctr.set_zoom(0.7)
        
            
            vis.poll_events()
            vis.update_renderer()
        
            # Capture image
            image = vis.capture_screen_float_buffer(do_render=True)
            filename = f"{output_prefix}_{axis}.png"
            o3d.io.write_image(filename, o3d.geometry.Image((255*np.asarray(image)).astype(np.uint8)))
            print(f"✅ Screenshot saved: {filename}")
        
        if keep_window_open:
            vis.run()
        
        vis.destroy_window()





class BPS_visualiser():
    def __init__(self, bps, mesh_res=6, output_filepath='output/', test_flag='poly', use_original_mesh=False, show_on_coarse=False, blend_type='default', just_onering = False ):

        self.mesh_res=mesh_res
        self.use_original_mesh=use_original_mesh
        self.show_on_coarse=show_on_coarse
        self.blend_type=blend_type
        
        
        
        self.bps=bps
        self.output_filepath = output_filepath

        if use_original_mesh == True:
            self.load_original_base_patches()
        else:
            self.load_regular_base_patches()


        if just_onering==True:

            # If you want indices from a onering
            self.select_patch_indices = torch.tensor(
                self.bps.onerings[0]['triangles'], dtype=torch.long, device=bps.device
            )
        else:
            # If you want all triangle indices
            self.select_patch_indices = torch.arange(
                self.bps.F.shape[0], dtype=torch.long, device=bps.device
            )


        self.test_flag=test_flag
        self.diffmod = DifferentialModule()



        self.meancurv_cmap = plt.get_cmap('seismic')
        self.meancurv_mapping = mapping12

         
        
        self.gausscurv_cmap = plt.get_cmap('seismic')
        self.gausscurv_mapping = mapping12

        self.facepatch_cmap = plt.get_cmap('tab20')

        self.angle_cmap = plt.get_cmap('hsv')
        #self.angle_cmap = plt.get_cmap('binary')

        self.error_mapping = linear100
        self.error_cmap = plt.get_cmap('Reds')

        

    def load_regular_base_patches(self):
        self.base_facepatches = [ o3d.io.read_triangle_mesh('data/high_precision_subdiv_triangles/triangle_'+str(self.mesh_res)+'.obj')
                             for i in range(self.bps.num_facepatches) ]
        return

    def load_original_base_facepatches(self):

        
        
        self.base_facepatches = [ o3d.io.read_triangle_mesh( self.bps.facepatches_path + '/' + str(i)+'_param.obj')
                             for i in range(self.bps.num_facepatches) ]

        self.base_facepatches_target = []

        max_num_points = max(len(np.asarray(p.vertices)) for p in self.base_facepatches)

        
        
        for facepatch in self.base_facepatches:
            uv = vertex_uvs_from_face_uvs(np.asarray(facepatch.triangles), np.asarray(facepatch.triangle_uvs))
            
            # Pad UVs to (N, 3) by adding a 0 z-column
            uv_xyz = np.pad(uv, ((0, 0), (0, 1)), mode='constant')
            
            # Pad rows to (max_num_points, 3)
            uv_padded = np.pad(uv_xyz, ((0, max_num_points - uv.shape[0]), (0, 0)), mode='constant')
        
            
        
            target_padded = np.pad(np.asarray(facepatch.vertices), ((0, max_num_points - len(patch.vertices)), (0, 0)), mode='constant')

            self.base_facepatches_target.append(target_padded)

            facepatch.vertices = Vector3dVector(uv_padded)

        return




    def compute_quantities(self, settings=['default'], degree=None):
        """
        Vectorised version of compute_quantities.
    
        This version attempts to do as much as possible in batch:
          - stacks domain samples for all patches -> wrt_batch (P, N, 2)
          - uses bns.forward output_points (P, N, 3) already produced above
          - calls diffmod methods on the whole batch where possible
        If diffmod.* functions do not accept batched inputs, it falls back to a per-patch loop (original behaviour).
        """
    
        facepatch_faces = [np.asarray(patch.triangles) for patch in self.base_facepatches]
        num_facepatches = len(self.bps.F)
    
        # build per-patch domain samples and pad to same length (P, N, 2)
        domain_samples_per_facepatch = [torch.tensor(np.asarray(facepatch.vertices)[:, :2]).float().to(self.bps.device) for facepatch in self.base_facepatches]
        max_pts = max(t.shape[0] for t in domain_samples_per_facepatch)
        wrt = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_pts - t.shape[0])) for t in domain_samples_per_facepatch])


        wrt.requires_grad = True

        precomputed_data = self.bps.precompute_data_from_samples(wrt, detached=False)


        if degree ==  None:
            degree = self.bps.degree

        print('degree:', degree)

        if 'unblended' in settings:
            output_points, unblended_points = self.bps.forward( precomputed_data, degree=degree, return_unblended=True )
        else:
            output_points = self.bps.forward( precomputed_data, degree=degree )

        print('output pts', output_points.shape)
        print('wrt', wrt.shape)

       
        
        # output_points shape expected: (P, N, 3)
        # convert to numpy for later meshing
        output_np = output_points.detach().cpu().numpy()
        self.output_np = output_np
    
        n_points_per_patch = wrt.shape[1]
    
        # extra data from precomputed_data dictionary
        self.blends = precomputed_data['blend_weights'].detach().cpu()
        self.angles = precomputed_data['angles'].detach().cpu()
        self.radii = precomputed_data['radii'].detach().cpu()
        
        self.bary = barycentric_coordinates(wrt[:, :, 0], wrt[:, :, 1]).squeeze().detach().cpu().numpy()
        #self.per_vertex_unblended = extra_data['per_vertex_unblended']
    
        verts = self.bps.V[self.bps.F].transpose(2, 1)
        coarse_embedding = np.einsum('bij,bjn->bni', verts.cpu(), self.bary)
    
        # containers
        # Use device consistent with output_points
        device = output_points.device
        dtype = output_points.dtype
    
        all_normals = torch.zeros((num_facepatches, n_points_per_patch, 3), dtype=dtype, device=device)
        all_meancurv = torch.zeros((num_facepatches, n_points_per_patch), dtype=dtype, device=device)
        all_gausscurv = torch.zeros((num_facepatches, n_points_per_patch), dtype=dtype, device=device)
        all_directions = torch.zeros((num_facepatches, n_points_per_patch, 3, 2), dtype=dtype, device=device)
        all_princ_curvatures = torch.zeros((num_facepatches, n_points_per_patch, 2), dtype=dtype, device=device)
        all_elastic_energy = torch.zeros((num_facepatches, n_points_per_patch), dtype=dtype, device=device)
    
        facepatch_indices = None
        if facepatch_indices is None:
            facepatch_indices = list(range(num_facepatches))
    
        # Try batch-mode differential ops. If diffmod's methods don't accept batch, fall back.
        
        # Normals (batch)
        if ('normals' in settings) or ('abs-normals' in settings):
            print("computing normals...")
            normals_batch = self.diffmod.compute_normals(out=output_points, wrt=wrt)  # expect (P,N,3)
            normals_batch = normals_batch.reshape(num_facepatches, n_points_per_patch, 3).to(dtype)
            all_normals = normals_batch.clone()


            ################# special treatment of exactly-on-vertex points #######################
            

            for i in facepatch_indices:
                
                for t in range(3):
                    poly_mat = self.bps.get_poly_matrix(self.bps.F[i][t])

                    grad_x = poly_mat [1,:]
                    grad_y = poly_mat [2,:]
    
                    cross_prod = torch.cross(grad_x, grad_y)
                    norm = cross_prod.norm(p=2, dim=-1, keepdim=True)
                    new_normal = cross_prod / norm
    
                    mask =  ( ((wrt - self.bps.base_triangle_verts[t]).abs() < 0.01).all(dim=-1) ).squeeze()
                    
                    rotated_normal = (new_normal @ self.bps.rotations[self.bps.F[i][t]]).to(dtype)
                    all_normals[i, mask[i,:], :] = rotated_normal
            
            
            ############################################################################################
            self.normals = all_normals
    

            

        # Curvatures (batch)
        if 'curv' in settings:
            print("Attempting batched curvature computation...")
            if 'directions' in settings:
                H, K, directions, princ_curvatures, normals = self.diffmod.compute_curvature(out=output_points, wrt=wrt, compute_principal_directions=True)
                
                
            else:
                H, K = self.diffmod.compute_curvature(out=output_points, wrt=wrt, compute_principal_directions=False)
                directions=None
                princ_curvatures=None
       


                    ################# special treatment of exactly-on-vertex points #######################
        
        for i in facepatch_indices:
            for t in range(3):
                # --- First derivatives (gradients at origin) ---
                # convention: poly_coeffs[k,:] = coefficient for derivative order k
                #   [0,:] constant term
                #   [1,:] derivative wrt u
                #   [2,:] derivative wrt v
                #   [3,:] second derivative wrt uu
                #   [4,:] second derivative wrt uv
                #   [5,:] second derivative wrt vv

                poly_mat = self.bps.get_poly_matrix(self.bps.F[i][t])
                #print('poly mat shape', poly_mat.shape)
                zeros = torch.zeros(10, poly_mat.shape[1], device=poly_mat.device, dtype=poly_mat.dtype)
                
                # Concatenate
                poly_mat = torch.cat([poly_mat, zeros], dim=0)


        
                r_u  = poly_mat[1, :]   # (3,)
                r_v  = poly_mat[2, :]   # (3,)

        
                # --- Cross product and area element ---
                N = torch.cross(r_u, r_v)            # unnormalized normal
                Delta = (N**2).sum()                 # = |N|^2
                norm = Delta.sqrt()
                n = N / norm                         # unit normal
        
                # --- Second derivatives at origin ---
                r_uu = 2 * poly_mat[3, :]   # d²r/du²
                r_uv =     poly_mat[4, :]   # d²r/dudv
                r_vv = 2 * poly_mat[5, :]   # d²r/dv²
        
                # --- First fundamental form coefficients ---
                E = (r_u * r_u).sum()
                F = (r_u * r_v).sum()
                G = (r_v * r_v).sum()
        
                # --- Second fundamental form coefficients ---
                L = (r_uu * n).sum()
                M = (r_uv * n).sum()
                Ncoef = (r_vv * n).sum()
        
                # --- Curvatures ---
                K_val = (L * Ncoef - M**2) / (E * G - F**2 + 1e-12)
                H_val = (E * Ncoef - 2 * F * M + G * L) / (2 * (E * G - F**2 + 1e-12))
        
                # --- Masking and assignment ---
                mask = (( (wrt - self.bps.base_triangle_verts[t]).abs() < 0.01).all(dim=-1)).squeeze()
           

        if 'curv' in settings:
            K[i, mask[i, :]] = K_val
            H[i, mask[i, :]] = H_val

            self.directions = directions
            self.princ_curvatures = princ_curvatures
            self.meancurv = H.squeeze()
            self.gausscurv = K.squeeze()
        
        #######################################################################################
        
     
            
        # ----------------------------
        # Build Open3D meshes (vectorized where easy)
        # ----------------------------
        facepatches = []
        coarse_facepatches = []
        unblended_facepatches = []
    
        for i in range(num_facepatches):
            output_V = output_np[i, :, :]
            facepatch = o3d.geometry.TriangleMesh()
            facepatch.vertices = Vector3dVector(output_V)
            facepatch.triangles = Vector3iVector(facepatch_faces[i])
            facepatch.compute_vertex_normals()
            facepatches.append(facepatch)
            #o3d.io.write_triangle_mesh(f"{self.output_filepath}output_patch_{i+1}.obj", patch)
    
            coarse_facepatch = o3d.geometry.TriangleMesh()
            coarse_facepatch.vertices = Vector3dVector(coarse_embedding[i, :, :])
            coarse_facepatch.triangles = Vector3iVector(facepatch_faces[i])
            coarse_facepatch.compute_vertex_normals()
            coarse_facepatches.append(coarse_facepatch)
    
        # Unblended patches coloring: vectorized color computation per patch (still loop over patches but not vertices)


        if 'unblended' in settings:
            
        
            for facepatch_id in range(self.bps.F.shape[0]):
                verts_idx = self.bps.F[facepatch_id]
                for k in range(3):
                    unblended_facepatch = o3d.geometry.TriangleMesh()
                    unblended_facepatch.vertices = Vector3dVector(unblended_points[facepatch_id, k, :, :].detach().cpu().numpy())
                    unblended_facepatch.triangles = Vector3iVector(facepatch_faces[facepatch_id] )
        
                    #blend = self.blends[facepatch_id, verts_idx[k], :]
                    #base_color = np.array(self.facepatch_cmap(facepatch_id % 20)[:-1])
                    #colors = np.ones((blend.shape[0], 1)) @ base_color[np.newaxis, :]
                    #unblended_facepatch.vertex_colors = Vector3dVector(colors.astype(np.float32))
                    unblended_facepatch.compute_vertex_normals()
                    unblended_facepatches.append(unblended_facepatch)
                
        
    
        # Error metric (if requested)
        if 'error' in settings:
            if not self.use_original_mesh:
                raise ValueError("Sorry, you need to specify << use_original_mesh=True >> if you want to show the error colourmap.")
            self.per_point_error = []
            for i in range(self.bns.num_facepatches):
                target = self.base_patches_target[i]
                mesh_verts = np.asarray(patches[i].vertices)
                self.per_point_error.append(np.sqrt(((target - mesh_verts) ** 2).sum(-1)))
    
        # attach results to object
        self.facepatches = facepatches
        self.coarse_facepatches = coarse_facepatches
        self.unblended_facepatches = unblended_facepatches
    





        
        #########################################

    def show_cbar(self, mapping, cmap, filename="output/colorbar.png"):
    
        values = mapping(np.linspace(0, 1, 256), invert=True)
        sm = ScalarMappable(cmap=cmap)
        sm.set_array(values)
    
        fig, ax = plt.subplots(figsize=(1, 5))
        fig.colorbar(sm, cax=ax)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    def show_implicit_error_on_pcd(self, shape_id, num_samples=100000 ):
        
        #visual_sample_dict = self.bps.compute_samples(num_samples=num_samples)
        
        #visual_t = visual_sample_dict['uv']
        #visual_output = self.bps.forward(visual_t, test_flag=self.test_flag, select_patch_indices=self.select_patch_indices).detach()
        
        
        
        #error = udf( visual_output, shape_id=shape_id)
        #colors = self.error_cmap(self.error_mapping( error ))[:,:,:3]
        
        #self.show_cbar(self.error_mapping, self.error_cmap, filename='output/error_cbar.png')

        visual_output = self.output_np
        colors = None
    
        show_f_list([{'points':visual_output, 'colors':colors}])   



    #def show_pcd_bps(self):
    #    show_f_list({})

    def show_bps(self, settings=['default'], vertex_id=0, patch_id=0, output_dir = 'rendering_resu/', show_on_coarse=False):
        self.show_on_coarse = show_on_coarse

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        facepatch_sets = [self.facepatches]
        actual_vertex_positions = self.bps.actual_vertex_positions()
        radius = 0.01
        
        actual_vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().cpu().numpy())
            .paint_uniform_color([1, 0, 0])
            for pos in actual_vertex_positions
        ]

        vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().cpu().numpy())
            .paint_uniform_color([0, 0, 1])
            for pos in self.bps.V
        ]

        vtx_blobs=[]
        actual_vtx_blobs=[]


        
        if self.show_on_coarse == True:
            facepatch_sets.append(self.coarse_facepatches)

        for facepatches in facepatch_sets:

            

            ############## DEFAULT VIEW (patches) ####################
            if 'default' in settings:
                for i,facepatch in enumerate(facepatches):
                    facepatch.paint_uniform_color(self.facepatch_cmap(i%20)[:-1])
        
                name = 'Output Mesh Resolution ' +str(self.mesh_res)
                #o3d.visualization.draw_geometries(patches, window_name=name)
                save_mesh_screenshots(facepatches+vtx_blobs+actual_vtx_blobs, output_prefix=str(output_dir)+'/patch-coloured')

                # Initialize an empty mesh
                combined_mesh = o3d.geometry.TriangleMesh()
                
                # Merge all meshes
                for facepatch in facepatches:
                    combined_mesh += facepatch
                
                
                # Save to file (e.g., .ply or .obj)
                o3d.io.write_triangle_mesh( str( output_dir / "patch-coloured.obj" ), combined_mesh)
                




            if 'one-colour' in settings:
                for i,facepatch in enumerate(facepatches):
                    facepatch.paint_uniform_color(np.array([0.5, 0.5, 1.0]))
                    #patch.paint_uniform_color(np.array([0.2, 0.0, 0.0]))
        
                name = 'Output Mesh Resolution ' +str(self.mesh_res)
                o3d.visualization.draw_geometries(facepatches+vtx_blobs+actual_vtx_blobs, window_name=name)
                save_mesh_screenshots(facepatches+vtx_blobs+actual_vtx_blobs, output_prefix=str(output_dir)+'/uniform-coloured')

                # Initialize an empty mesh
                combined_mesh = o3d.geometry.TriangleMesh()
                
                # Merge all meshes
                k=0
                for facepatch in facepatches:
                    combined_mesh += facepatch
                    #o3d.io.write_triangle_mesh("output/patch_"+str(k)+".obj", facepatch) #write individual patches, optionally
                    k+=1
                
            
                # Save to file (e.g., .ply or .obj)
                o3d.io.write_triangle_mesh( str(output_dir / "uniform-coloured.obj" ), combined_mesh)
                
    
            if 'normals' in settings:
                # ------------------------------------------------------------
                # Color facepatches (Open3D path – unchanged)
                # ------------------------------------------------------------
                all_vertices = []
                all_faces = []
                all_normals_colours = []
            
                v_offset = 0
            
                for i, facepatch in enumerate(facepatches):
                    # Vertex colors for Open3D visualization
                    
                    colors = (0.5 * self.normals[i, :, :].detach().cpu().numpy() + 0.5).clip(0.0, 1.0)
                    #colors = (np.abs( self.normals[i, :, :].detach().cpu().numpy()) ).clip(0.0, 1.0)
                    
                    facepatch.vertex_colors = Vector3dVector(colors)
            
                    # Collect vertices and faces for export
                    V_patch = np.asarray(facepatch.vertices)
                    F_patch = np.asarray(facepatch.triangles)
                    normals_colours_patch = colors
            
                    all_vertices.append(V_patch)
                    all_faces.append(F_patch + v_offset)  # offset for merged mesh
                    all_normals_colours.append(normals_colours_patch)
            
                    v_offset += V_patch.shape[0]
            
                # Merge all vertices/faces/colors
                V = np.vstack(all_vertices)
                F = np.vstack(all_faces)
                normals_colours = np.vstack(all_normals_colours)
            
                # ------------------------------------------------------------
                # Merge meshes for Open3D OBJ export
                # ------------------------------------------------------------
                combined_mesh = o3d.geometry.TriangleMesh()
                for facepatch in facepatches:
                    combined_mesh += facepatch
            
                o3d.visualization.draw_geometries(
                    facepatches + vtx_blobs + actual_vtx_blobs,
                    window_name="Normals"
                )
            
                # Save OBJ (as before)
                o3d.io.write_triangle_mesh(
                    str(output_dir / "normals-coloured.obj"),
                    combined_mesh
                )
            
                # ============================================================
                # Hakowan-compatible PLY using your function
                # ============================================================
                combined_mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(combined_mesh.vertices),
                faces=np.asarray(combined_mesh.triangles),
                process=False
                )
                
                # Scalar/Vector dictionary
                colouringdict = {"normals_colours": normals_colours }
                
                # Write Hakowan-compatible PLY
                write_custom_colour_ply_file(
                    tm=combined_mesh_trimesh,
                    colouringdict=colouringdict,
                    filepath=str("rendering/rendering_results/normals.ply")
                )




            if 'abs-normals' in settings:
                    # ------------------------------------------------------------
                    # Color facepatches (Open3D path – unchanged)
                    # ------------------------------------------------------------
                    all_vertices = []
                    all_faces = []
                    all_normals_colours = []
                
                    v_offset = 0
                
                    for i, facepatch in enumerate(facepatches):
                        # Vertex colors for Open3D visualization
                        
                        #colors = (0.5 * self.normals[i, :, :].detach().cpu().numpy() + 0.5).clip(0.0, 1.0)
                        colors = (np.abs( self.normals[i, :, :].detach().cpu().numpy()) ).clip(0.0, 1.0)
                        
                        facepatch.vertex_colors = Vector3dVector(colors)
                
                        # Collect vertices and faces for export
                        V_patch = np.asarray(facepatch.vertices)
                        F_patch = np.asarray(facepatch.triangles)
                        normals_colours_patch = colors
                
                        all_vertices.append(V_patch)
                        all_faces.append(F_patch + v_offset)  # offset for merged mesh
                        all_normals_colours.append(normals_colours_patch)
                
                        v_offset += V_patch.shape[0]
                
                    # Merge all vertices/faces/colors
                    V = np.vstack(all_vertices)
                    F = np.vstack(all_faces)
                    normals_colours = np.vstack(all_normals_colours)
                
                    # ------------------------------------------------------------
                    # Merge meshes for Open3D OBJ export
                    # ------------------------------------------------------------
                    combined_mesh = o3d.geometry.TriangleMesh()
                    for facepatch in facepatches:
                        combined_mesh += facepatch
                
                    o3d.visualization.draw_geometries(
                        facepatches + vtx_blobs + actual_vtx_blobs,
                        window_name="Normals"
                    )
                
                    # Save OBJ (as before)
                    o3d.io.write_triangle_mesh(
                        str(output_dir / "normals-coloured.obj"),
                        combined_mesh
                    )
                
                    # ============================================================
                    # Hakowan-compatible PLY using your function
                    # ============================================================
                    combined_mesh_trimesh = trimesh.Trimesh(
                    vertices=np.asarray(combined_mesh.vertices),
                    faces=np.asarray(combined_mesh.triangles),
                    process=False
                    )
                    
                    # Scalar/Vector dictionary
                    colouringdict = {"abs-normals_colours": normals_colours }
                    
                    # Write Hakowan-compatible PLY
                    write_custom_colour_ply_file(
                        tm=combined_mesh_trimesh,
                        colouringdict=colouringdict,
                        filepath=str("rendering/rendering_results/mobius-abs-normals.ply")
                    )


                        

            ################ CURVATURE VIEWS ###################
            if 'curv' in settings:
                for i, facepatch in enumerate(facepatches):
                    
                    colors = self.meancurv_cmap( self.meancurv_mapping ( self.meancurv[i,:].detach().cpu().numpy() ) )[:,:-1]
    
                    facepatch.vertex_colors = Vector3dVector(colors)
                    
                
                o3d.visualization.draw_geometries(facepatches+vtx_blobs+actual_vtx_blobs, window_name='Mean Curvature')
                #self.show_cbar(self.meancurv_mapping, self.meancurv_cmap, filename='output/meancurv_cbar.png')
    
    
                for i, facepatch in enumerate(facepatches):
                    
                    colors = self.gausscurv_cmap( self.gausscurv_mapping ( self.gausscurv[i,:].detach().cpu().numpy() ) )[:,:-1]
    
                    facepatch.vertex_colors = Vector3dVector(colors)
                    
                
                o3d.visualization.draw_geometries(facepatches+vtx_blobs+actual_vtx_blobs, window_name='Gauss Curvature')
                #self.show_cbar(self.gausscurv_mapping, self.gausscurv_cmap, filename='output/gausscurv_cbar.png')



            if 'directions' in settings:
                print('Directions shape:', self.directions.shape)
            
                # Paint all patches uniformly white
                for facepatch in facepatches:
                    facepatch.paint_uniform_color((1.0, 1.0, 1.0))
            
                # Create LineSets for both min and max curvature directions
                mincurv_arrows = []
                maxcurv_arrows = []
            
                # Iterate through each patch and its corresponding direction set
                for facepatch_idx, patch in enumerate(facepatches):
                    verts = np.asarray(facepatch.vertices)
                    n = verts.shape[0]
            
                    for direction_idx, color in enumerate([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]):  # Red for min, blue for max
                        directions = self.directions[direction_idx, facepatch_idx, :, :].squeeze().detach().cpu().numpy()

                        #print((directions*directions).sum(-1))
                        directions = np.nan_to_num(directions, nan=0.0)  # Clean up NaNs
            
                        # Scale for visibility
                        scale_factor = 0.01
                        endpoints = np.concatenate((verts, verts + scale_factor * directions))
            
                        # Create LineSet
                        arrow_set = o3d.geometry.LineSet()
                        arrow_set.points = Vector3dVector(endpoints)
                        arrow_set.lines = Vector2iVector(np.array([[i, i + n] for i in range(n)]))
                        arrow_set.paint_uniform_color(color)

                        if direction_idx==0:
                            mincurv_arrows.append(arrow_set)
                        else:
                            maxcurv_arrows.append(arrow_set)
            
                # Visualize everything at once
                o3d.visualization.draw_geometries( mincurv_arrows + maxcurv_arrows, window_name='Curvature Directions')
                
                o3d.visualization.draw_geometries(facepatches + mincurv_arrows, window_name='Minimum Curvature Directions')
                o3d.visualization.draw_geometries(facepatches + maxcurv_arrows, window_name='Maximum Curvature Directions')
                o3d.visualization.draw_geometries(facepatches + mincurv_arrows + maxcurv_arrows, window_name='Curvature Directions')



            if 'param' in settings:

                print('unblended', np.array(self.unblended_facepatches[0].vertices).shape, 'usual', np.array(self.coarse_facepatches[0].vertices).shape)
                # --- Use coarse facepatch vertices to define coloring ---
                coarse_coords_list = [np.asarray(coarse.vertices) for coarse in self.coarse_facepatches]
                param_colors_list = []

                from matplotlib import colormaps

                stripe_cmap = colormaps["ocean"]
            
                for coarse_coords in coarse_coords_list:
                    # Example: striped coloring based on y-coordinate
                    
                    #colors = np.zeros_like(coarse_coords)

                    phase = ( 0.5*np.tanh(np.sin(coarse_coords[:,2]*100)) + 0.5 ) * 0.8

                    colors = stripe_cmap(phase)[:,:3]
                    
                    param_colors_list.append(colors)

                print('facepatches,', len(self.unblended_facepatches), len(self.coarse_facepatches))
            
                # --- Apply coarse coloring to all facepatches ---
                for i, facepatch in enumerate(facepatches):
                    # Map coarse coloring to fine patch vertices via barycentric embedding
                    coarse_colors = param_colors_list[i]  # shape: (N_coarse, 3)
                    facepatch.vertex_colors = Vector3dVector(coarse_colors)

                    self.coarse_facepatches[i].vertex_colors = Vector3dVector(coarse_colors)
                    self.unblended_facepatches[3*i].vertex_colors = Vector3dVector(coarse_colors)
                    self.unblended_facepatches[3*i + 1].vertex_colors = Vector3dVector(coarse_colors)
                    self.unblended_facepatches[3*i + 2].vertex_colors = Vector3dVector(coarse_colors)
            
                
                combined_unblended_mesh = o3d.geometry.TriangleMesh()
                for unblended_facepatch in self.unblended_facepatches:
                    combined_unblended_mesh += unblended_facepatch
                
                combined_unblended_mesh_trimesh = trimesh.Trimesh(
                    vertices=np.asarray(combined_unblended_mesh.vertices),
                    faces=np.asarray(combined_unblended_mesh.triangles),
                    process=False
                    )
                                    

                save_mesh_screenshots(self.unblended_facepatches+vtx_blobs+actual_vtx_blobs, output_prefix=str(output_dir) + "/unblended-param")


                
            
                # Save screenshots
                
                save_mesh_screenshots(self.coarse_facepatches  + vtx_blobs + actual_vtx_blobs,
                                      output_prefix=str(output_dir) + "/coarse-param")
                
                save_mesh_screenshots(facepatches  + vtx_blobs + actual_vtx_blobs,
                                      output_prefix=str(output_dir) + "/bs-param")
            
                # --- Save PLY files for both coarse and fine patches ---
                combined_fine_mesh = o3d.geometry.TriangleMesh()
                for patch in facepatches:
                    combined_fine_mesh += patch
                combined_coarse_mesh = o3d.geometry.TriangleMesh()
                for patch in self.coarse_facepatches:
                    combined_coarse_mesh += patch

                param_colours = np.asarray(combined_fine_mesh.vertex_colors)

                unblended_param_colours = np.asarray(combined_unblended_mesh.vertex_colors)

                unblended_colouringdict = {"param_colours": unblended_param_colours } 
                #colouringdict = {"patch_colours": }


                # Write Hakowan-compatible PLY
                write_custom_colour_ply_file(
                    tm=combined_unblended_mesh_trimesh,
                    colouringdict=unblended_colouringdict,
                    filepath=str("rendering/rendering_results/param_unblended.ply")
                )
            
                # Write colored PLYs
                write_custom_colour_ply_file(
                    tm=trimesh.Trimesh(np.asarray(combined_fine_mesh.vertices),
                                       np.asarray(combined_fine_mesh.triangles),
                                       process=False),
                    colouringdict={"param_colours": param_colours},
                    filepath=str("rendering/rendering_results/param_fine.ply")
                )
            
                write_custom_colour_ply_file(
                    tm=trimesh.Trimesh(np.asarray(combined_coarse_mesh.vertices),
                                       np.asarray(combined_coarse_mesh.triangles),
                                       process=False),
                    colouringdict={"param_colours": param_colours },
                    filepath=str("rendering/rendering_results/param_coarse.ply")
                )

                
                


    
            if 'angle' in settings:

                angle_colors = self.angle_cmap(( 1*self.angles/(2*np.pi))  )[:,:,:,:-1]
                print('angle colours', angle_colors.shape)
                print('blend', self.blends.shape)
                blended_angle_colors = (self.blends.unsqueeze(-1) * angle_colors ).sum(-2)

                circle_angle_colors = ( ( self.radii.unsqueeze(-1)<0.5) * angle_colors ).sum(-2)

                print('radii shape', self.radii.shape)
                

                k=0
                for facepatch in facepatches:
                    facepatch.vertex_colors = Vector3dVector(blended_angle_colors[k, :,:])
                    
                    #o3d.io.write_triangle_mesh("output/facepatch_"+str(k)+".obj", facepatch) #write individual patches, optionally
                    k+=1

                save_mesh_screenshots(facepatches+vtx_blobs+actual_vtx_blobs, output_prefix='output/angle')

                combined_mesh = o3d.geometry.TriangleMesh()
                for patch in facepatches:
                    combined_mesh += patch

                combined_coarse_mesh = o3d.geometry.TriangleMesh()
                for patch in self.coarse_facepatches:
                    combined_coarse_mesh += patch


                angle_colours = np.asarray( combined_mesh.vertex_colors )
                
                write_custom_colour_ply_file(
                    tm=trimesh.Trimesh(np.asarray(combined_mesh.vertices),
                                       np.asarray(combined_mesh.triangles),
                                       process=False),
                    colouringdict={"angle_colours": angle_colours },
                    filepath=str("rendering/rendering_results/angle.ply")
                )

                write_custom_colour_ply_file(
                    tm=trimesh.Trimesh(np.asarray(combined_coarse_mesh.vertices),
                                       np.asarray(combined_coarse_mesh.triangles),
                                       process=False),
                    colouringdict={"angle_colours": angle_colours },
                    filepath=str("rendering/rendering_results/coarse-angle.ply")

                )


      


            if 'blend' in settings:


                '''


                colors = np.zeros((self.bps.F.shape[0], self.blends.shape[1], 3))

                for i, v in enumerate(self.bps.F[patch_id]):
                    for f in range(self.bps.F.shape[0]):
                        if v in self.bps.F[f]:
                            j = np.where(self.bps.F[f] == v)[0][0]   # local index in face f
                            colors[f, :, i] += self.blends[f, :, j]
                '''


                colors = np.zeros((self.bps.F.shape[0], self.blends.shape[1], 3))

                print(colors.shape, self.blends.shape)
                verts = self.bps.F[patch_id,:]

                for f in range(self.bps.F.shape[0]):
                    for i, v in enumerate(self.bps.F[f]):
                        if verts[0] == v :
                            colors[f, :, 0] += self.blends[f, :, i].detach().cpu().numpy()

                        if verts[1] == v :
                            colors[f, :, 1] += self.blends[f, :, i].detach().cpu().numpy()
                        if verts[2] == v :
                            colors[f, :, 2] += self.blends[f, :, i].detach().cpu().numpy()
                        
                        

                for f, facepatch in enumerate(facepatches):
                    facepatch.vertex_colors = Vector3dVector(colors[f,:,:])

                save_mesh_screenshots(facepatches+vtx_blobs+actual_vtx_blobs, output_prefix="output/blend")

                #write_custom_colour_ply_file(
                #    tm=combined_mesh_trimesh,
                #    colouringdict=colouringdict,
                #    filepath=str("rendering/rendering_results/unblended.ply")
                #)

                #save_mesh_screenshots(self.unblended_facepatches+vtx_blobs+actual_vtx_blobs, output_prefix="output/blend")


                

            
            if 'discrete_blend' in settings:
                for i, facepatch in enumerate(patches):
    
                    colors = self.blends[i, self.bps.F[patch_id,:], : ].transpose(1,0)
                    colors = np.floor(colors*10.0)/10.0
                    
                    facepatch.vertex_colors = Vector3dVector(colors)
                    
                save_mesh_screenshots(facepatches+vtx_blobs+actual_vtx_blobs, output_prefix="output/discrete-blend")

                
    
            if 'bary' in settings:
                for i, facepatch in enumerate(facepatches):
    
                    colors = self.bary[i,:,:].transpose(1,0)
                    
                    facepatch.vertex_colors = Vector3dVector(colors)
                    
                save_mesh_screenshots(facepatches+vtx_blobs+actual_vtx_blobs, output_prefix="output/bary")


            if 'unblended' in settings:

                combined_mesh = o3d.geometry.TriangleMesh()
                for facepatch in self.unblended_facepatches:
                    combined_mesh += facepatch
                
                combined_mesh_trimesh = trimesh.Trimesh(
                    vertices=np.asarray(combined_mesh.vertices),
                    faces=np.asarray(combined_mesh.triangles),
                    process=False
                    )
                    
                colouringdict = {"patch_colours": np.repeat(normals_colours, 3, axis=0) }    #just temporary. but looks cool.
                #colouringdict = {"patch_colours": }

                
                    
                # Write Hakowan-compatible PLY
                write_custom_colour_ply_file(
                    tm=combined_mesh_trimesh,
                    colouringdict=colouringdict,
                    filepath=str("rendering/rendering_results/unblended.ply")
                )

                save_mesh_screenshots(self.unblended_facepatches+vtx_blobs+actual_vtx_blobs, output_prefix="output/unblended")
                


            if 'error' in settings:
                for i, facepatch in enumerate(facepatches):
    
                    colors = self.error_cmap(self.error_mapping(self.per_point_error[i]))[:,:3]

                    
                    patch.vertex_colors = Vector3dVector(colors)
                    
                o3d.visualization.draw_geometries(facepatches, window_name='Position Error Per Point (Distance, not squared)')
                self.show_cbar(self.error_mapping, self.error_cmap, filename='output/error_cbar.png')
                


class Elastic_energy_visualiser():
    def __init__(self, bns1, bns2, mesh_res=6, output_filepath='output/', test_flag=None, use_original_mesh=False, show_on_coarse=False, blend_type='default', just_onering=False,
                alpha=1.0, beta=1.0, energy_visual_scale=1.0 ):

        self.mesh_res=mesh_res
        self.use_original_mesh=use_original_mesh
        self.show_on_coarse=show_on_coarse
        self.blend_type=blend_type
        
        self.bns1=bns1
        self.bns2=bns2
        self.output_filepath = output_filepath
        self.alpha=alpha
        self.beta=beta
        

        if use_original_mesh == True:
            self.load_original_base_patches()
        else:
            self.load_regular_base_patches()


        if just_onering==True:

            # If you want indices from a onering
            self.select_patch_indices = torch.tensor(
                self.bns2.onerings[0]['triangles'], dtype=torch.long, device=self.bns2.device
            )
        else:
            # If you want all triangle indices
            self.select_patch_indices = torch.arange(
                self.bns2.F.shape[0], dtype=torch.long, device=self.bns2.device
            )
            
        self.test_flag=test_flag

        self.diffmod = DifferentialModule()


        self.elastic_energy_cmap = plt.get_cmap('Oranges')
        self.elastic_energy_mapping = lambda x: linear(x, factor=energy_visual_scale) 

        self.area_dist_cmap = plt.get_cmap('BrBG')
        self.area_dist_mapping = lambda x: 0.4*np.log(x) + 0.5
        #self.area_dist_mapping = lambda x: linear(x, factor=1e2) 


        
        

    def load_regular_base_patches(self):
        self.base_patches = [ o3d.io.read_triangle_mesh('data/subdiv_triangles/triangle_'+str(self.mesh_res)+'.obj')
                             for i in range(self.bns1.num_facepatches) ]
        return

    def load_original_base_patches(self):

        
        self.base_patches = [ o3d.io.read_triangle_mesh( self.bns1.patches_path + '/' + str(i)+'_param.obj')
                             for i in range(self.bns1.num_facepatches) ]

        self.base_patches_target = []



        max_num_points = max(len(np.asarray(p.vertices)) for p in self.base_patches)
        self.base_patches_target = []
        
        for patch in self.base_patches:
            uv = vertex_uvs_from_face_uvs(np.asarray(patch.triangles), np.asarray(patch.triangle_uvs))
            
            # Pad UVs to (N, 3) by adding a 0 z-column
            uv_xyz = np.pad(uv, ((0, 0), (0, 1)), mode='constant')
            
            # Pad rows to (max_num_points, 3)
            uv_padded = np.pad(uv_xyz, ((0, max_num_points - uv.shape[0]), (0, 0)), mode='constant')
        
            
        
            target_padded = np.pad(np.asarray(patch.vertices), ((0, max_num_points - len(patch.vertices)), (0, 0)), mode='constant')

            self.base_patches_target.append(target_padded)

            patch.vertices = Vector3dVector(uv_padded)
            
        return
        


    def compute_quantities(self, precomputed_data):
        # Get faces for each patch
        patch_faces = [np.asarray(patch.triangles) for patch in self.base_patches]
        num_facepatches = len(self.bns1.F)
    
        # Collect domain samples (2D coords per patch)
        domain_samples_per_patch = [
            torch.tensor(np.asarray(patch.vertices)[:, :2])
            for patch in self.base_patches
        ]
        for patch in domain_samples_per_patch:
            patch.requires_grad = True
    
        # Pad samples so all patches align in size
        max_points = max(t.shape[0] for t in domain_samples_per_patch)
        domain_samples = torch.stack([
            torch.nn.functional.pad(
                t, (0, 0, 0, max_points - t.shape[0])
            )
            for t in domain_samples_per_patch
        ])  # shape: (num_facepatches, max_points, 2)

        
        subprocess.run(
                [
                    "python", "precomputation.py",
                    "--output-filepath", "data/precomputation_results/temp-samples.pth",
                    "--config-filepath", config_filepath
                ],
                check=True
            )
        precomputed_training_data = torch.load('data/precomputation_results/temp-samples.pth')
    
        # Forward pass through both networks
        output_points1 = self.bns1.forward(
            domain_samples
        )
        output_points2 = self.bns2.forward(
            domain_samples
        )
    
        output_2_np = output_points2.detach().cpu().numpy()
        output_1_np = output_points1.detach().cpu().numpy()
        
        n_points_per_patch = domain_samples.shape[1]
    
    
        W, total_energy, FFF = self.diffmod.elastic_energy_density(out1=output_points1, out2=output_points2, wrt=domain_samples, alpha=self.alpha, beta=self.beta)
    
        # Store per-patch elastic energies
        self.elastic_energy = W
        
        self.area_dist = torch.sqrt ( FFF[:,:,0,0] * FFF[:,:,1,1] - FFF[:,:,1,0]**2 )

        
        # Rebuild meshes for visualization and compute area ratios
        deformed_patches = []
        rest_patches = []
        triangle_area_ratios = []
        
        vertex_area_ratios = []
        
        for i in range(num_facepatches):
            # --- Rest patch ---
            rest_V = output_1_np[i, :, :]
            rest_patch = o3d.geometry.TriangleMesh()
            rest_patch.vertices = Vector3dVector(rest_V)
            rest_patch.triangles = Vector3iVector(patch_faces[i])
            rest_patch.compute_vertex_normals()
            rest_patches.append(rest_patch)
        
            # --- Deformed patch ---
            def_V = output_2_np[i, :, :]
            deformed_patch = o3d.geometry.TriangleMesh()
            deformed_patch.vertices = Vector3dVector(def_V)
            deformed_patch.triangles = Vector3iVector(patch_faces[i])
            deformed_patch.compute_vertex_normals()
            deformed_patches.append(deformed_patch)
        
            # --- Vectorised area computation ---
            tris = np.asarray(patch_faces[i])  # (F, 3)
        
            # Gather triangle vertices
            v0r, v1r, v2r = rest_V[tris[:, 0]], rest_V[tris[:, 1]], rest_V[tris[:, 2]]
            v0d, v1d, v2d = def_V[tris[:, 0]], def_V[tris[:, 1]], def_V[tris[:, 2]]
        
            # Cross products and norms → triangle areas
            rest_areas = 0.5 * np.linalg.norm(np.cross(v1r - v0r, v2r - v0r), axis=1)  # (F,)
            def_areas  = 0.5 * np.linalg.norm(np.cross(v1d - v0d, v2d - v0d), axis=1)  # (F,)
        
            # --- Distribute triangle areas to vertices (1/3 each) ---
            n_vertices = rest_V.shape[0]
            rest_vertex_areas = np.zeros(n_vertices)
            def_vertex_areas  = np.zeros(n_vertices)
        
            np.add.at(rest_vertex_areas, tris.ravel(), np.repeat(rest_areas / 3.0, 3))
            np.add.at(def_vertex_areas,  tris.ravel(), np.repeat(def_areas / 3.0, 3))
        
            # --- Compute per-vertex ratios ---
            ratios = np.divide(def_vertex_areas, rest_vertex_areas,
                               out=np.zeros_like(def_vertex_areas),
                               where=rest_vertex_areas > 1e-12)
            vertex_area_ratios.append(ratios)
        
        self.deformed_patches = deformed_patches
        self.rest_patches = rest_patches
        self.vertex_area_ratios = np.stack(vertex_area_ratios)   # list of (Vi,) arrays
        

    def show_cbar(self, mapping, cmap, filename="output/colorbar.png"):
    
        values = mapping(np.linspace(0, 1, 256), invert=True)
        sm = ScalarMappable(cmap=cmap)
        sm.set_array(values)
    
        fig, ax = plt.subplots(figsize=(1, 5))
        fig.colorbar(sm, cax=ax)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
  

    def show_bns(self, settings=['default'], vertex_id=0, patch_id=0):

        

        patch_sets = [self.deformed_patches]
        if self.show_on_coarse == True:
            patch_sets.append(self.coarse_patches)


        actual_vertex_positions = self.bns2.actual_vertex_positions()
        radius = 0.01
        
        actual_vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().numpy())
            .paint_uniform_color([1, 0, 0])
            for pos in actual_vertex_positions
        ]

        vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().numpy())
            .paint_uniform_color([0, 0, 1])
            for pos in self.bns2.V
        ]

        for patches in patch_sets:
            
            
            for i, patch in enumerate(patches):
                
                colors = self.area_dist_cmap( self.area_dist_mapping ( self.area_dist[i,:].detach().numpy() ) )[:,:-1]

                patch.vertex_colors = Vector3dVector(colors)
            
            save_mesh_screenshots(patches+vtx_blobs+actual_vtx_blobs, output_prefix="output/cts_area_dist")
            #o3d.visualization.draw_geometries(patches, window_name='cts differential area distortion')

            '''


            for i, patch in enumerate(patches):
                
                colors = self.area_dist_cmap( self.area_dist_mapping ( self.vertex_area_ratios[i,:] ) )[:,:-1]

                patch.vertex_colors = Vector3dVector(colors)
                
            o3d.visualization.draw_geometries(patches, window_name='discrete (per onering) area distortion')

            
            for i, patch in enumerate(patches):

                colors = ( 0.5 * self.normals[i,:,:].detach().numpy() + 0.5).clip(0,1)
                patch.vertex_colors = Vector3dVector(colors)

            o3d.visualization.draw_geometries(patches, window_name='normals')
            '''

            
            for i, patch in enumerate(patches):
                
                colors = self.elastic_energy_cmap( self.elastic_energy_mapping ( self.elastic_energy[i,:].detach().numpy() ) )[:,:-1]

                patch.vertex_colors = o3d.utility.Vector3dVector(colors)
                
            #o3d.visualization.draw_geometries(patches, window_name='elastic energy')
            save_mesh_screenshots(patches+vtx_blobs+actual_vtx_blobs, output_prefix='elastic energy')
        


        data = np.log10( self.elastic_energy.flatten().detach().cpu().numpy() )
        #data = data[(data >= -20)]
        plt.figure()
        plt.hist(data, bins=100)
        plt.title("Elastic Energy Distribution (log10) ")
        plt.show()


        






'''
# deprecated: for fixed correspondences

def show_error_on_pcd(bns, num_samples=100000, include_edges=False):
    visual_sample_dict = bns.compute_samples(num_samples=num_samples, analytic=True, analytic_shape='torus', include_edges=include_edges)
    
    visual_t = visual_sample_dict['uv']
    visual_output = bns.forward(visual_t).detach().numpy()
    
    visual_target = visual_sample_dict['target'].numpy()
        
    error = np.sqrt(((visual_output - visual_target)**2).sum(-1))
    
    pcd_visualiser = BNS_visualiser(bns, mesh_res=2, use_original_mesh=False)
    
    colors = pcd_visualiser.error_cmap(pcd_visualiser.error_mapping( error ))[:,:,:3]
    
    
    pcd_visualiser.show_cbar(pcd_visualiser.error_mapping, pcd_visualiser.error_cmap, filename='output/error_cbar.png')
    
    show_f_list([{'points':visual_output, 'colors':colors}])
'''














'''


############# Showing B function (used for radial and 2d_pou) and visualising the overlap #############


x = torch.arange(0.0, 1.01, 0.01)

# Triangle vertices
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0.5, np.sqrt(3)/2])
vertices = [A, B, C]
r = (1+overlap_param)/2

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# First plot: B(x) and B(1 - x)
ax1.plot(x, bns.B(x), label='B(x)')
ax1.plot(x, bns.B(1 - x), label='B(1 - x)')
ax1.set_title("Function Plot")
ax1.legend()
ax1.grid(True)

# Second plot: triangle and circles
triangle = plt.Polygon(vertices, fill=None, edgecolor='black')
ax2.add_patch(triangle)
for v in vertices:
    circle_nonzero = plt.Circle(v, r, fill=False, color='blue', linestyle='--')
    ax2.add_patch(circle_nonzero)
    circle_unit = plt.Circle(v, r-overlap_param, fill=False, color='green', linestyle='--')
    ax2.add_patch(circle_unit)

ax2.set_aspect('equal')
ax2.set_xlim(-r, 1 + r)
ax2.set_ylim(-r, 1)
ax2.set_title("Triangle with Circles")
ax2.grid(True)

plt.tight_layout()
plt.show()

'''













