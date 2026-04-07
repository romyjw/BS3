import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from bns_utils import *
from mesh_processing import *
import os
import trimesh

from scipy.interpolate import CubicSpline

import torch.nn.functional as F


from differential import * 
diffmod = DifferentialModule()

two_pi = 2*torch.pi
pi = torch.pi

print('Using open3d version at',o3d.__path__)


class BPS_fast(nn.Module):
    def __init__(self, surface_config = None, device='cpu', deformed=False):
        super(BPS_fast, self).__init__()

        print('Initialising BPS on', device)

        self.device = device

        self.blend_type = surface_config['blend_type']
        if deformed==False:
            self.coarse_patches_id = surface_config['coarse_patches_id']
        else:
            self.coarse_patches_id = surface_config['coarse_patches_id']+'-deformed'
            
        self.angle_flag = 'equal' #no longer used
        self.overlap_param = surface_config['overlap_param']    # 0.15470053837 < v < 0.73205080756
        self.degree = surface_config['degree'] #polynomial degree
        self.global_scale = surface_config['global_scale'] # scaling of all the polynomials (after evaluating polynomials)
        self.local_scales = surface_config['local_scales'] # true of false (if true, also make polynomial scaling proportional to local edge-lengths)

        #if 'equivariance' in surface_config.keys():
        #    self.equivariance = surface_config['equivariance']
        #else:
        equivariance = True
        if equivariance == False:
            print('Warning!! Equivariance is turned off.')

        
        filepath = 'data/surfaces/'+self.coarse_patches_id+'.obj'
        coarse_mesh_o3d = o3d.io.read_triangle_mesh(filepath)
        self.coarse_mesh_tm = trimesh.load(filepath)

        


        self.base_triangle_verts = torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, torch.sqrt(torch.tensor(3.0, dtype=torch.float32, device=self.device)) / 2],
                ],
                dtype=torch.float32,
                device=self.device,
            )


        
        
        self.he_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(coarse_mesh_o3d)
        print(self.he_mesh)
        
        #halfedge stuff for the coarse representation
        self.V = torch.tensor(np.asarray(self.he_mesh.vertices), dtype=torch.float32, device=self.device)
        self.n = self.V.shape[0]
        self.F = np.asarray(self.he_mesh.triangles)
        self.he = self.he_mesh.half_edges

        

        self.num_facepatches = self.F.shape[0]

        #vertex normals of coarse mesh
        triangle_mesh = o3d.geometry.TriangleMesh()
        triangle_mesh.vertices = self.he_mesh.vertices
        triangle_mesh.triangles = self.he_mesh.triangles
        triangle_mesh.compute_vertex_normals()
        self.vertex_normals = np.array(triangle_mesh.vertex_normals)



        # V: (n, 3) torch tensor of vertices
        # F: (num_faces, 3) numpy array of vertex indices
        
        V = self.V                      # (n, 3), torch
        F = torch.as_tensor(self.F, device=self.device)  # (F, 3)
        
        n = V.shape[0]
        
        # Accumulators
        edge_sum = torch.zeros(n, device=self.device)
        edge_count = torch.zeros(n, device=self.device)
        
        # Edges per triangle
        edges = [(0, 1), (1, 2), (2, 0)]
        
        for i, j in edges:
            vi = F[:, i]
            vj = F[:, j]
        
            lengths = torch.norm(V[vi] - V[vj], dim=1)  # (F,)
        
            edge_sum.index_add_(0, vi, lengths)
            edge_sum.index_add_(0, vj, lengths)
        
            edge_count.index_add_(0, vi, torch.ones_like(lengths))
            edge_count.index_add_(0, vj, torch.ones_like(lengths))
        
        # Mean edge length per vertex
        mean_edge_length_per_vertex = edge_sum / edge_count.clamp(min=1)
        mean_edge_length = mean_edge_length_per_vertex.mean()

        print('mean edge length', mean_edge_length)

        if self.local_scales == True:
            print('Using Local Scales')
            self.local_scales = mean_edge_length_per_vertex[F]  # (F, 3)
        else:
            print('No local scale: using same scale for all vertex functions.')
            self.local_scales = mean_edge_length * torch.ones((F.shape[0], 3), device=self.device) # (F, 3)


        self.disp_modifier = None

        # a halfedge is assigned to each vertex (onering)
        print('Assiging a halfedge to each vertex.')
        initial_V_he = get_V_he(self.he_mesh)
        
        self.V_he = optimise_V_he(self.he_mesh, initial_V_he) #Important if we want to do boundaries! Allows us to find onerings of boundaries.

    
        print('base triangle verts', self.base_triangle_verts.dtype)

        # --- Remap OBJ sharp edges -> Open3D indices, then find halfedges ---
        
        # Parse OBJ vertices and sharp edges (OBJ indices)
        obj_vertices = self.coarse_mesh_tm.vertices
        
        obj_sharp_edges = parse_sharp_edges_from_obj(filepath)  # 0-based (i, j)

        #obj_sharp_edges = [(self.F[i][0], self.F[i][1]) for i in range(self.F.shape[0])] #test
        # Open3D vertices
        o3d_vertices = np.asarray(self.he_mesh.vertices)
        
        # Position -> Open3D index map (rounded for numerical safety)
        def vkey(v):
            return tuple(np.round(v, 5))
        
        pos_to_o3d = {vkey(v): i for i, v in enumerate(o3d_vertices)}
        self.pos_to_o3d=pos_to_o3d

        #print('pos to o3d', pos_to_o3d)
        
        # Remap sharp edges into Open3D index space
        sharp_edges_o3d = []
        for i, j in obj_sharp_edges:
            vi = vkey(obj_vertices[i])
            vj = vkey(obj_vertices[j])
            #print('vi,vj', vi,vj)
            if vi in pos_to_o3d.keys() and vj in pos_to_o3d.keys():
                sharp_edges_o3d.append((pos_to_o3d[vi], pos_to_o3d[vj]))

        print('sharp edges obj', obj_sharp_edges)
        print('sharp edges o3d', sharp_edges_o3d)
        
        # Match halfedges
        sharp_edge_set = {frozenset(e) for e in sharp_edges_o3d}
        
        self.sharp_halfedges = [
            idx for idx, he in enumerate(self.he_mesh.half_edges)
            if -1 not in he.vertex_indices
            and frozenset(he.vertex_indices) in sharp_edge_set
        ]
        
        print("sharp halfedges:", self.sharp_halfedges)

        

        self.onerings = compute_onering_data(self.he_mesh, self.V_he, vtx_normals=self.vertex_normals,            sharp_halfedges=self.sharp_halfedges)

        print('Computing rotations.')
        
        
        self.compute_fixed_rotations(equivariance = equivariance)
        
        
        #print('Warning: will not be equivariant because local coord frames could not be properly generated.')
        #self.rotations = [torch.eye() for i in range(self.V.shape[0])]

        self.default_coarse_points = self.V
        self.default_rotations = self.rotations

        '''
        print('Cannot use halfedge mesh so defaulting to a usual mesh')
        
        #halfedge stuff for the coarse representation
        self.V = torch.tensor(np.asarray(coarse_mesh_o3d.vertices), dtype=torch.float32, device=self.device)
        self.n = self.V.shape[0]
        self.F = np.asarray(coarse_mesh_o3d.triangles)
        

        self.num_facepatches = self.F.shape[0]


        self.vertex_normals = np.array(coarse_mesh_o3d.vertex_normals)
        '''    
    

        
        self.set_poly_coeffs_to_default(degree=self.degree)
        

        

        


        self.test_scale = 1.0


    def poly_basis(self, X, Y, degree):
        """
        Generate basis terms in the exact same order as the original P lambdas:
        [1, X, Y, X², XY, Y², X³, X²Y, XY², Y³, ...]
        """
        terms = [torch.ones_like(X)]
        for d in range(1, degree + 1):
            for i in range(d, -1, -1):  # X^i Y^(d-i)
                j = d - i
                terms.append((X ** i) * (Y ** j))
        return torch.stack(terms, dim=1)


    def poly_basis_gradient(self, X, Y, degree):
        """
        Generate gradient of the above polynomial basis, wrt X and Y:
        [1, X, Y, X², XY, Y², X³, X²Y, XY², Y³, ...]
            -->
        [0, 1, 0, 2X, Y, 0, 3X^2, 2XY, Y^2, 0, ...]
        [0, 0, 1, 0, X, 2Y,  0,   X^2, 2XY, 3Y^2, ...]
        """
        dBx = [torch.zeros_like(X)]  # ∂/∂X of 1
        dBy = [torch.zeros_like(Y)]  # ∂/∂Y of 1
    
        for d in range(1, degree + 1):
            for i in range(d, -1, -1):  # X^i Y^(d-i)
                j = d - i
    
                # ∂/∂X
                if i == 0:
                    dBx.append(torch.zeros_like(X))
                else:
                    dBx.append(i * (X ** (i - 1)) * (Y ** j))
    
                # ∂/∂Y
                if j == 0:
                    dBy.append(torch.zeros_like(Y))
                else:
                    dBy.append(j * (X ** i) * (Y ** (j - 1)))
    
        dBx = torch.stack(dBx, dim=1)
        dBy = torch.stack(dBy, dim=1)
    
        # stack last dimension as (d/dX, d/dY)
        return torch.stack([dBx, dBy], dim=-1)
    


    def set_poly_coeffs_to_default(self, degree=3):
        size = int((degree + 1) * (degree + 2) / 2)
        V = self.V.shape[0]
    
        # Create one big tensor instead of many small ones
        coeffs = torch.zeros(V, size, 3, device=self.device)
    
        # Default plane: linear terms X and Y become identity
        # term index 1 = du, term index 2 = dv in your current layout
        if size > 1:
            coeffs[:, 1, 0] = 1.0  # ∂/∂u → affects x
        if size > 2:
            coeffs[:, 2, 1] = 1.0  # ∂/∂v → affects y
    
        # This is now ONE Parameter
        self.poly_coeffs = nn.Parameter(coeffs)


    def get_poly_matrix(self, vtx_idx, degree=None):
        if degree is None:
            degree = self.degree
    
        num_terms = int((degree + 1) * (degree + 2) / 2)
        return self.poly_coeffs[vtx_idx, :num_terms, :]  # (num_terms, 3)

    def get_whole_poly_tensor(self, degree=None):
        if degree is None:
            degree = self.degree
    
        num_terms = int((degree + 1) * (degree + 2) / 2)
        return self.poly_coeffs[:, :num_terms, :]
        
    def reset(self):
        self.rotations = self.default_rotations
        self.V = self.default_coarse_points
        self.scaling_weights = [1 for i in range(self.n)]

    def compute_fixed_rotations(self, equivariance=True):
        self.rotations = []

        if equivariance==True:
            for i in range(self.n):
                # vertex normal as float32 on the correct device
                normal = torch.tensor(self.vertex_normals[i], dtype=torch.float32, device=self.device)
            
                # chosen neighbour vertex
                v = self.he[self.V_he[i]].vertex_indices[1]
            
                # direction vector in float32 on device
                dir_0 = (self.V[v, :] - self.V[i, :]).to(dtype=torch.float32, device=self.device)
            
                # orthogonalize against normal
                dir_0 = dir_0 - (dir_0 * normal).sum() * normal
            
                # normalize
                dir_0 = dir_0 / torch.sqrt((dir_0 * dir_0).sum())
            
                # build rotation matrix (still float32, on device)
                R = torch.stack(
                    [dir_0,
                     -1.0 * torch.cross(normal, dir_0, dim=0),
                     normal]
                )
            
                self.rotations.append(R)
        else:
            self.rotations = [torch.eye(3, device = self.device) for i in range(self.n)]




    def compute_samples(self, num_samples=10000, avoid_corners=False, eps=0.02):
        """
        Sample points on mesh faces.
    
        Args:
            num_samples (int): number of samples per face.
            avoid_corners (bool): if True, resample points too close to vertices.
            eps (float): minimum distance from any corner in barycentric coords.
        """
        embedded_triangles = []
        patches_bary = []
        uvs = []
        for triangle_index in range(self.F.shape[0]):
    
            if not avoid_corners:
                # Regular barycentric sampling
                u = torch.rand(num_samples, 1, device=self.device)
                v = torch.rand(num_samples, 1, device=self.device)
                sqrt_u = torch.sqrt(u)
                bary = torch.cat([
                    1 - sqrt_u,
                    sqrt_u * (1 - v),
                    sqrt_u * v
                ], dim=1)
            else:
                # Rejection + resampling loop
                bary = torch.empty((0, 3), device=self.device)
                while bary.shape[0] < num_samples:
                    u = torch.rand(num_samples, 1, device=self.device)
                    v = torch.rand(num_samples, 1, device=self.device)
                    sqrt_u = torch.sqrt(u)
                    candidates = torch.cat([
                        1 - sqrt_u,
                        sqrt_u * (1 - v),
                        sqrt_u * v
                    ], dim=1)
    
                    mask = (candidates > eps).all(dim=1)
                    accepted = candidates[mask]
    
                    bary = torch.cat([bary, accepted], dim=0)
    
                # Trim to exactly num_samples
                bary = bary[:num_samples]
    
            patches_bary.append(bary)
    
            verts = self.V[self.F[triangle_index, :]]
            embedded_triangles.append(bary @ verts)
            uvs.append(bary @ self.base_triangle_verts.float())
    
        embedded_triangles = torch.stack(embedded_triangles)
        patches_bary = torch.stack(patches_bary)
        uvs = torch.stack(uvs)
    
        sample_dict = {
            'bary': patches_bary,
            'coarse_embedding': embedded_triangles,
            'uv': uvs
        }
    
        return sample_dict



    def bary_weight(self, x, i):
        v_a = x - self.base_triangle_verts[i].to(x.device)
        v_b = x - self.base_triangle_verts[(i + 1) % 3].to(x.device)
        v_c = x - self.base_triangle_verts[(i + 2) % 3].to(x.device)
    
        # Compute 2D cross product as scalar
        cross = v_b[:,0] * v_c[:,1] - v_b[:,1] * v_c[:,0]
    
        bary_weight = 0.5 * abs(cross.to(x.device)) / (0.5 * 0.5 * np.sqrt(3))
    
        return bary_weight

    def discrete_weight(self, x, i):
        v_a = x - self.base_triangle_verts[i]
        side_vec1 = self.base_triangle_verts[(i+1)%3] - self.base_triangle_verts[i]
        side_vec2 = self.base_triangle_verts[(i+2)%3] - self.base_triangle_verts[i]

        dot1 = (v_a*side_vec1).sum(-1)
        dot2 = (v_a*side_vec2).sum(-1)

        discrete_weight = torch.logical_and(dot1 <= 0.5, dot2 <= 0.5).float()

        return discrete_weight


    def get_batch(self, precomputed_data, idx):
        """
        Differentiable batching: 
        Extracts batch elements by index without breaking autograd.
        """
    
        batch_dict = {}
    
        # ensure idx is a proper tensor index
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=next(iter(precomputed_data.values())).device)
        elif isinstance(idx, list) or isinstance(idx, tuple):
            idx = torch.tensor(idx, device=next(iter(precomputed_data.values())).device)
    
        for key, data in precomputed_data.items():
            #print(key)
    
            # Keep dimensions consistent — indexing must preserve batch axes.
            # DO NOT wrap with torch.tensor(), which kills gradients.
    
            if key == 'poly_basis':
                # poly_basis is shaped e.g. [B, ?, N, ...]
                # We select the N dimension via idx
                batch_dict[key] = data[:, :, idx, ...]   # keeps autograd
                
            elif key == 'poly_basis_gradient':
                # poly_basis is shaped e.g. [B, ?, N, ...]
                # We select the N dimension via idx
                batch_dict[key] = data[:, :, idx, ...]   # keeps autograd
    
            elif key in ('rotations', 'per_face_vertices', 'all_J_proxy', 'all_Jc_pinv'):
                # these do NOT depend on idx — but do not recreate tensors
                batch_dict[key] = data   # pass through untouched
            
    
            else:
                # generic case: shape [B, N, ...]
                batch_dict[key] = data[:, idx, ...]      # keeps autograd
    
        return batch_dict


    


    def precompute_data_from_samples(self, x, detached=True):
        print('Precomputing blend weights etc, using', self.blend_type, 'blending.')

        if detached==False:
            x.requires_grad=True




        all_J_proxy = torch.zeros((self.F.shape[0], 2, 2), device=self.device)   # F x 2 x 2
        all_Jc_pinv = torch.zeros((self.F.shape[0], 2, 3), device=self.device)   # F x 2 x 3

        
        

        all_onering_coords = torch.zeros((self.F.shape[0], x.shape[1], 3, 2), device=self.device) # F x S x 3 x 2
        all_blend_weights = torch.zeros((self.F.shape[0], x.shape[1], 3), device=self.device) # F x S x 3
        all_angles = torch.zeros((self.F.shape[0], x.shape[1], 3), device=self.device) # F x S x 3
        all_radii = torch.zeros((self.F.shape[0], x.shape[1], 3), device=self.device) # F x S x 3

        
        all_onering_coords_gradients = torch.zeros((self.F.shape[0], x.shape[1], 3, 2, 2 ), device=self.device ) # F x S x 3 x 2 x 2 
        all_blend_weights_gradients = torch.zeros((self.F.shape[0], x.shape[1], 3, 2), device=self.device) # F x S x 3 x 2
            
    
        for face_index in range(self.F.shape[0]): #loop over all faces

            if detached==True:

                cur_x = x[face_index].clone().detach().unsqueeze(0).requires_grad_(True)
            else:
                cur_x = x[face_index].unsqueeze(0)

            #print('curx', cur_x.shape)

            verts = self.F[face_index, :] #loop over the 3 vertex indices for current face

            p_inv = torch.tensor([
                    [1.0, -1.0 / math.sqrt(3)],
                    [0.0,  2.0 / math.sqrt(3)]
                ], device=self.device)
            
            #Q = torch.stack([self.V[verts[1]] - self.V[verts[0]], self.V[verts[2]] - self.V[verts[0]] ], dim=1)  # (3,2)
            #J = Q @ p_inv  # (3,2)
            
            all_J_proxy[face_index, :, :] = p_inv

            # edges in R^3
            e1 =  self.V[verts[1]] - self.V[verts[0]]
            e2 =  self.V[verts[2]] - self.V[verts[0]]
            # J_c : (3,2)
            Jc = torch.stack([e1, e2], dim=-1)
            # Optional: regularization for near-degenerate triangles
            eps = 1e-10
            # Pseudoinverse: (2,3)
            # Jc : (..., 3, 2)
            JT = Jc.transpose(-1, -2)           # (..., 2, 3)
            G  = JT @ Jc                         # (..., 2, 2)
            a = G[..., 0, 0]
            b = G[..., 0, 1]
            c = G[..., 1, 1]
            det = a * c - b * b
            det = det + eps                     # regularize
            invG = torch.stack([
                torch.stack([ c / det, -b / det], dim=-1),
                torch.stack([-b / det,  a / det], dim=-1),
            ], dim=-2)                          # (..., 2, 2)
            Jc_pinv = invG @ JT                  # (..., 2, 3)

            all_Jc_pinv[face_index, :, :] = Jc_pinv

            

            for i in range(3):
            
                onering = self.onerings[verts[i]]

                j = onering['triangles'].index(face_index)


                

                flip_func = lambda x: x
                
                '''
                if (face_index in [104, 105, 202]):   #### crazy hardcoding just for mobius band
                    if verts[i]==21:
                        flip_func=lambda x: pi - x
                    elif verts[i]==65:
                        
                        flip_func = lambda x: -x + 5 * two_pi/onering['valence']
                    elif verts[i]==18:
                        
                        flip_func = lambda x: -x +  two_pi/onering['valence']

                elif (face_index==203 and verts[i]==18):
                    flip_func = lambda x: - x  + 7 * two_pi/onering['valence']
                '''

                
                
                onering_x, radii, theta, new_theta = self.onering_coords(cur_x, i, j, onering['valence'], flag=self.angle_flag, disc_index=onering['sharp_halfedges'], flip_func=flip_func)

                blend_weight = self.blend_weight(cur_x, radii, i, j, blend_type=self.blend_type)
                r_a, r_b, r_c = radii

                
                

                all_onering_coords[face_index, :, i, :] = onering_x
                all_blend_weights[face_index, :, i] = blend_weight
                all_angles[face_index, :, i] = new_theta
                all_radii[face_index,:,i] = radii[i%3]



                if detached==True:
                    #print('cur x grad?', cur_x.requires_grad)
                    blend_weight_gradient = diffmod.gradient(out=blend_weight.reshape((1,-1,1)), wrt = cur_x, allow_unused=True ).squeeze()
                    #print('bw grad', blend_weight_gradient.shape)
                    onering_x_gradient = diffmod.gradient(out=onering_x.reshape(1, -1,2), wrt = cur_x, allow_unused=True)
                    #print('oc grad', onering_x_gradient.shape)
    
                    all_onering_coords_gradients[face_index, :, i, :, :] = onering_x_gradient
                    all_blend_weights_gradients[face_index, :, i, :] = blend_weight_gradient
                

  

            
        torch.mps.synchronize()
        print('Now deadling with poly basis')
                
        XY = 1.0 * all_onering_coords
        X = XY[:,:,:,0]
        Y = XY[:,:,:, 1]
        
        poly_basis = self.poly_basis(X, Y, degree=self.degree).to(self.device)

        poly_basis_gradient = 1.0 * self.poly_basis_gradient(X, Y, degree=self.degree).to(self.device) # compute gradient of poly basis wrt onering coords analytically (they're polynomials so it's easy)

        rotations = torch.stack(self.rotations)[self.F,:,:].to(self.device)
        per_face_vertices = self.V[self.F].unsqueeze(2)

        
                
        precomputed_data = {'equilateral_triangle_samples': x, 'onering_coords': all_onering_coords, 'blend_weights': all_blend_weights,
                            'poly_basis': poly_basis, 'rotations':rotations, 'per_face_vertices':per_face_vertices, 'radii':all_radii,
                            'angles':all_angles, 'all_J_proxy':all_J_proxy,    'all_Jc_pinv':all_Jc_pinv,
                            'onering_coords_gradient': all_onering_coords_gradients, 'blend_weights_gradient': all_blend_weights_gradients,
                           'poly_basis_gradient': poly_basis_gradient}

        if detached==True:
            for key, data in precomputed_data.items():
                precomputed_data[key] = data.detach()


        return precomputed_data

    

            
            
        


    def onering_coords(self, x, i, j, valence=6, flag='equal', disc_index=None, flip_func = (lambda x: x) ):
        
    
        
    
        # Ensure base_triangle_verts is on the same device
        v_a = x - self.base_triangle_verts[i].to(x.device)
        v_b = x - self.base_triangle_verts[(i + 1) % 3].to(x.device)
        v_c = x - self.base_triangle_verts[(i + 2) % 3].to(x.device)


        r_a = torch.sqrt((v_a ** 2).sum(-1))
        r_b = torch.sqrt((v_b ** 2).sum(-1))
        r_c = torch.sqrt((v_c ** 2).sum(-1))
    
        v = v_a
        r = r_a

    
        theta = torch.atan2(v[:,:, 1], v[:,:, 0])
        
        theta -= i * (two_pi / 3)
        
        
        theta = theta + (theta < 0) * two_pi
        theta = theta - (theta >= two_pi) * two_pi
        theta = two_pi / 6 - theta

        theta.clamp(0.0, two_pi/6)

        global_theta = ((theta + (j - 1) * two_pi / 6) * (6 / valence))
    
        if flag == 'equal':
            new_theta = global_theta
        else:
            raise Exception(f"{flag} is not a valid flag. It may have been deprecated.")
            
    
        new_theta = new_theta + (new_theta < 0) * two_pi
        new_theta = new_theta - (new_theta >= two_pi) * two_pi

        
        if not (disc_index is None or disc_index==[]):
            new_theta = add_discontinuity(new_theta, shift = (two_pi/valence) * disc_index[0] )  
        
        #if not (disc_index is None or disc_index==[]):
        #    new_theta = add_discontinuities(new_theta, disc_index=disc_index, valence=valence )  

        new_theta = flip_func(new_theta) % two_pi #### special feature required for non-orientable surfaces/ surfaces with an inconsistent orientation

        onering_x = torch.stack([r * torch.cos(new_theta), r * torch.sin(new_theta)],
                                dim=0).to(dtype=torch.float32, device=self.device).squeeze().transpose(1,0)
    
        return onering_x, (r_a, r_b, r_c), theta, new_theta




    def blend_weight(self, x, radii, i=0, j=0, face_index = 0, blend_type=None, overlap_param=None):
        if overlap_param is None:
            overlap_param = self.overlap_param

        r_a, r_b, r_c = radii
                
        if blend_type=='bary':
            bary_weight = self.bary_weight(x[face_index,:,:].squeeze(), i)
            blend_weight = bary_weight

        elif blend_type=='discrete':
            blend_weight = self.discrete_weight(x[face_index,:,:].squeeze(), i)
            
        elif blend_type=='radial_bump_func':
            blend_weight = B1_bump(r_a, v=overlap_param)

        elif blend_type=='pou_bump_func': #this one is a bit wobbly, inv exp is smoother looking
            blend_weight = B1_bump(r_a, v=overlap_param) / ( B1_bump(r_a, v=overlap_param) + B1_bump(r_b, v=overlap_param) + B1_bump(r_c, v=overlap_param)     )

        elif blend_type=='radial_inv_exp':
            blend_weight = B2_inv_exp(r_a, v=overlap_param)

        elif blend_type=='pou_inv_exp':
            blend_weight = B2_inv_exp(r_a, v=overlap_param) / ( B2_inv_exp(r_a, v=overlap_param) + B2_inv_exp(r_b, v=overlap_param) + B2_inv_exp(r_c, v=overlap_param)     )

        elif blend_type=='radial_trig': # use the trigonometric blending from Yuksel's curves paper
            blend_weight = B3_trig(r_a, v=overlap_param)

        elif blend_type=='pou_trig': # use the trigonometric blending from Yuksel's curves paper
            blend_weight = B3_trig(r_a, v=overlap_param) / ( B3_trig(r_a, v=overlap_param) + B3_trig(r_b, v=overlap_param) + B3_trig(r_c, v=overlap_param)     )

        elif blend_type=='pou_circular': # use the trigonometric blending from Yuksel's curves paper
            blend_weight = B4_circular(r_a, v=overlap_param) / ( B4_circular(r_a, v=overlap_param) + B4_circular(r_b, v=overlap_param) + B4_circular(r_c, v=overlap_param)     )

        elif blend_type=='simple_exp':
            blend_weight = B5_simple_exp(r_a, v=overlap_param)

        elif blend_type=='pou_simple_exp':
            blend_weight = B5_simple_exp(r_a, v=overlap_param)/ (B5_simple_exp(r_a, v=overlap_param)+B5_simple_exp(r_b, v=overlap_param)+B5_simple_exp(r_c, v=overlap_param))

        elif blend_type=='djuren':
            bary = [ self.bary_weight(x[face_index,:,:].squeeze(), k) for k in range(3) ]
            verts = self.F[face_index, :]
            V_face = torch.tensor(self.V[verts, :], dtype=torch.float32, device=x.device)   # (3,3)
            
            blend_weight = B_djuren( bary, V_face, verts, i, exponent=2, dist_exponent=2 )
            
            
        else:
            raise Exception('Sorry, did not understand this blend type:', blend_type)

        return blend_weight

    '''


    def save_poly_coeffs(self, filename="poly_coeffs.pth"):
        save_data = {
            v: [p.detach().cpu().numpy() for p in coeff_list]
            for v, coeff_list in enumerate(self.poly_coeffs)
        }
        torch.save(save_data, filename)


    
    def load_poly_coeffs(self, filename="poly_coeffs.pth"):
        poly_coeffs_data = torch.load(filename, map_location=self.device)
        self.poly_coeffs = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.tensor(vec, dtype=torch.float32, device=self.device))
                              for vec in coeff_list])
            for coeff_list in poly_coeffs_data.values()
        ])
    '''

    def save_poly_coeffs(self, filename="poly_coeffs.pth"):
        """
        Save the big (V, num_terms, 3) parameter tensor.
        """
        torch.save(
            self.poly_coeffs.detach().cpu(),   # store as plain tensor
            filename
        )

    def load_poly_coeffs(self, filename="poly_coeffs.pth"):
        """
        Load the big (V, num_terms, 3) tensor and wrap it as a nn.Parameter.
        """
        tensor = torch.load(filename, map_location=self.device)
    
        # Ensure correct shape
        if tensor.shape != self.poly_coeffs.shape:
            raise ValueError(
                f"Loaded coeffs of shape {tensor.shape}, "
                f"but current model expects {self.poly_coeffs.shape}."
            )
    
        self.poly_coeffs = nn.Parameter(tensor.to(self.device))



    def actual_vertex_positions(self):
        actual_vertex_positions = []
        for i in range(self.V.shape[0]):

            offset = self.get_poly_matrix(i)[0,:]
            cur_vtx_position =  offset @ self.rotations[i] + self.V[i]
    
            actual_vertex_positions.append(cur_vtx_position)
            
        return actual_vertex_positions


        
    
    def forward(self, precomputed_data, test_flag = None, select_patch_indices=None, return_unblended=False,
               degree = None):
     
        if degree is None: #if polynomial degree is unspecified, use the degree specified in the config file
            degree = self.degree

        b = int((degree+1)*(degree+2)/2) #number of basis elements required for specified polynomial degree

        #x = precomputed_data['equilateral_triangle_samples']
        
        poly_basis = precomputed_data['poly_basis'][:, :b,:, :] #truncate poly_basis if not full degree
        blend_weights = precomputed_data['blend_weights']
        rotations = precomputed_data['rotations']
        per_face_vertices = precomputed_data['per_face_vertices']
                    
        A = self.get_whole_poly_tensor(degree = degree).to(self.device)
    
        A_hat = A[self.F,:,:]

        
        #print(poly_basis.device, A.device, A_hat.device)
        PXY = torch.einsum('fbsp,fpbc->fpsc', poly_basis, A_hat) #fpsc means faces x perspectives(3) x samples x channels(3)
        scaled_PXY = torch.einsum('fp,fpsc->fpsc', self.local_scales, PXY)
        

        oriented_polys = self.global_scale * scaled_PXY @ rotations  +  per_face_vertices

        output = torch.einsum('fsp,fpsc->fsc', blend_weights, oriented_polys)

        #if self.disp_modifier is not None:
        #    bs_normals = diffmod.compute_normals(out=output, wrt=x)

        #    print('using disp modifier')
        #    old=output
        #    output = self.disp_modifier(output, bs_normals)

        if return_unblended==True:
            return output, oriented_polys
            
        return output










        