import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from bns_utils import *
from mesh_processing import *
import os
import trimesh
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

import torch.nn.functional as F
import copy

from differential import * 
diffmod = DifferentialModule()

two_pi = 2*torch.pi
pi = torch.pi

print('Using open3d version at',o3d.__path__)


class Djuren_surface():
    def __init__(self, surface_config = None, device='cpu', mesh_res = 4):

            
        self.load_regular_base_patch(mesh_res)
        print('Initialising Djuren surface on', device)

        self.device = device

        self.coarse_patches_id = surface_config['coarse_patches_id']
        
        filepath = 'data/surfaces/'+self.coarse_patches_id+'.obj'
        coarse_mesh_o3d = o3d.io.read_triangle_mesh(filepath)
        self.coarse_mesh_tm = trimesh.load(filepath)
        
        self.he_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(coarse_mesh_o3d)
        
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

        # a halfedge is assigned to each vertex (onering)
        print('Assiging a halfedge to each vertex.')
        initial_V_he = get_V_he(self.he_mesh)
        
        self.V_he = optimise_V_he(self.he_mesh, initial_V_he) #Important if we want to do boundaries! Allows us to find onerings of boundaries.


        self.base_triangle_verts = torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, torch.sqrt(torch.tensor(3.0, dtype=torch.float32, device=self.device)) / 2],
                ],
                dtype=torch.float32,
                device=self.device,
            )

        self.onerings = self.compute_onering_data(self.he_mesh, self.V_he)
        

    
    def load_regular_base_patch(self, mesh_res):
        self.base_facepatch = o3d.io.read_triangle_mesh('data/high_precision_subdiv_triangles/triangle_'+str(mesh_res)+'.obj')
                             
        self.x = torch.tensor( self.base_facepatch.vertices)[:,:2]
        return

    def compute_onering_data(self, he_mesh, V_he, vtx_normals=None, sharp_halfedges=[]):
        V = np.asarray(he_mesh.vertices)
        he = he_mesh.half_edges
        
        onerings = []
    
        for i in range(V.shape[0]):
    
            v_pos_central = V[i,:]
            
            onering = {'V_indices':[], 'V_pos':[], 'triangles':[], 'valence':0, 'cumulative_angles':[], 'central_pos': v_pos_central, 'sharp_halfedges':[], 'is_boundary': False }
            cur_he = he[V_he[i]]
            
            vtx_0 = cur_he.vertex_indices[1]
    
            vtx = vtx_0
    
            idx = 0
            while (vtx!=vtx_0 or idx==0) :
                if not vtx in onering['V_indices']:
                    onering['V_indices'].append(vtx)
                    
                onering['triangles'].append(cur_he.triangle_index)
    
                v_pos_cur = V[vtx,:]
    
                onering['V_pos'].append(v_pos_cur)
    
                temp_he = he[cur_he.next].next
                onering['V_indices'].append(he[temp_he].vertex_indices[0])
    
                new_he_index = he[temp_he].twin
                idx+=1
    
                if new_he_index == -1:
                    onering['is_boundary'] = True
                    break
                else:
                    
                    cur_he = he[ new_he_index ]
                    vtx = cur_he.vertex_indices[1]
                    
                
    
                if new_he_index in sharp_halfedges:
                    onering['sharp_halfedges'].append(idx)
                
            valence=idx

            onering['V_indices'] = onering['V_indices'][::-1]
            onering['triangles'] = onering['triangles'][::-1]
            #onering['triangles'] = onering['triangles'][-1:] + onering['triangles'][:-1]
    
            onering['valence'] = valence
    
            center_idx = i   
            v_center = self.V[center_idx]
            
            cumulative_angle = 0.0
            onering['cumulative_angles'] = []
            
            for i in range(valence):
                v1_idx = onering['V_indices'][i]
                v2_idx = onering['V_indices'][(i+1) % valence]
            
                v_1 = self.V[v1_idx] - v_center   # edge vector
                v_2 = self.V[v2_idx] - v_center   # next edge
            
                # Normalize (important for stability)
                v_1 = v_1 / (torch.norm(v_1) + 1e-8)
                v_2 = v_2 / (torch.norm(v_2) + 1e-8)
            
                # Dot product → angle
                dot = torch.clamp(torch.dot(v_1, v_2), -1.0, 1.0)
                cur_angle = torch.acos(dot)
            
                cumulative_angle += cur_angle.item()
                onering['cumulative_angles'].append(cumulative_angle)
    
            
            onerings.append(onering)

        return onerings
   

    def compute_blend_weights(self):
    
        exponent = 2
    
        x = self.x #equilateral

        
        bary = torch.stack([self.bary_weight(x, i) for i in range(3)]).transpose(1, 0)
    
        dtype = bary.dtype
        device = bary.device
    
        blend_weights = torch.zeros(self.F.shape[0], x.shape[0], 3, dtype=dtype, device=device)
        self.Xb = torch.zeros(self.F.shape[0], x.shape[0], 3, dtype=dtype, device=device)
        
        for face_index in range(self.F.shape[0]):
            verts = self.F[face_index, :]
            #print(bary.shape, verts.shape)
    
            V_face = torch.tensor(self.V[verts, :], dtype=dtype, device=device)   # (3,3)
            Xb = bary @ V_face          # (num_samples, 3)
            self.Xb[face_index, :, :] = Xb
    
            for i in range(3):
            
                onering = self.onerings[verts[i]]
    
                j = onering['triangles'].index(face_index)

                eps = 1.0e-8
                denom = ( (bary[:,0]*bary[:,1])**exponent + (bary[:,1]*bary[:,2])**exponent + (bary[:,2]*bary[:,0])**exponent + eps)

                W_E_0 = ((bary[:,(i+1)%3]*bary[:,(i+2)%3])**exponent) / denom
    
                W_E_1 = ((bary[:,(i+0)%3]*bary[:,(i+2)%3])**exponent) / denom
                W_E_2 = ((bary[:,(i+0)%3]*bary[:,(i+1)%3])**exponent) / denom
                
                v_i = torch.tensor(self.V[verts[i], :], dtype=dtype, device=device)
                v_ip1 = torch.tensor(self.V[verts[(i+1)%3], :], dtype=dtype, device=device)
                v_ip2 = torch.tensor(self.V[verts[(i+2)%3], :], dtype=dtype, device=device)

        
                e_01 = ((Xb - v_ip1)**2).sum(-1)  /  ( ((Xb - v_i)**2).sum(-1) + ((Xb - v_ip1)**2).sum(-1) )  
                e_02 = ((Xb - v_ip2)**2).sum(-1)  /  ( ((Xb - v_i)**2).sum(-1) + ((Xb - v_ip2)**2).sum(-1) )  

                blend_weights[face_index, :, i] = e_01 * W_E_2 + e_02 * W_E_1
                
        self.blend_weights = blend_weights
        return blend_weights # faces x samples x 3
    
        
    def bary_weight(self, x, i):
        v_a = x - self.base_triangle_verts[i].to(x.device)
        v_b = x - self.base_triangle_verts[(i + 1) % 3].to(x.device)
        v_c = x - self.base_triangle_verts[(i + 2) % 3].to(x.device)
    
        # Compute 2D cross product as scalar
        cross = v_b[:,0] * v_c[:,1] - v_b[:,1] * v_c[:,0]
    
        bary_weight = 0.5 * abs(cross.to(x.device)) / (0.5 * 0.5 * np.sqrt(3))
    
        return bary_weight


    def forward(self, function_type = 'constant'):

   
        f,p,s,c = self.F.shape[0], 3, self.x.shape[0], 3

        
        
        onering_coords = torch.zeros(f,p,s,2)
        angles = torch.zeros(f,p,s)
        angles_eq = torch.zeros(f,p,s)

        for f_i in range(self.F.shape[0]):

            verts = self.V[self.F[f_i],:]
            for i in range(3):

                vtx_idx = self.F[f_i][i]
                
                onering = self.onerings[vtx_idx]
                
                j = onering['triangles'].index(f_i)
                
                cur_onering_coords, _,_,angle, angle_eq = self.onering_coords_func(self.x, i, j, vtx_idx, verts)
                onering_coords[f_i, i, :,:] =  cur_onering_coords
                angles[f_i, i, :] = angle
                angles_eq[f_i, i, :] = angle_eq

        

        
        self.onering_coords = onering_coords
        self.angles = angles
        self.angles_eq = angles_eq
        
        self.vertex_funcs = torch.zeros(f,p,s,c)

        if function_type == 'constant':
            for f_i in range(self.F.shape[0]):
                for k in range(3):
                    self.vertex_funcs[f_i, k, :,:] = self.V[self.F[f_i][k]]

        elif function_type == 'polynomial':

            poly_weights = []
            for vtx_idx in range(self.V.shape[0]):

                onering = self.onerings[vtx_idx]

                valence = onering['valence']
                
                
                P = torch.stack( [ self.onering_coords_func(torch.tensor([0.0,0.0]).unsqueeze(0),  1, (j+1)%valence, vtx_idx,   torch.zeros(3,3)   )[0]  for j in range(valence) ] ) #oc coords of onering vertices

                #print('P', P)
                
                Y = torch.stack(  [ self.V[onering['V_indices'][(j+1)%valence]] - self.V[vtx_idx,:]   for j in range(valence)  ] ) # vertex coords
                A = P.transpose(1,0)@P
                b = P.transpose(1,0)@Y

                #print ('A', A)
                W = torch.linalg.solve( A  ,   b   ) #solve Ax=b
                #print('W shape', W.shape)

                poly_weights.append(W)

            for f_i in range(self.F.shape[0]):
                for k in range(3):

                    vtx_idx = self.F[f_i][k]
                    onering = self.onerings[vtx_idx]

                    j = onering['triangles'].index(f_i)
                    valence = onering['valence']

                    
                    poly_basis = self.onering_coords_func(self.x,  k, j%valence, vtx_idx,   torch.zeros(3,3)   )[0]
                    #self.vertex_funcs[f_i, k, :,:] += self.blend_weights[f_i, k, :] * (poly_basis @ poly_weights[vtx_idx] + self.V[vtx_idx,:])
                    self.vertex_funcs[f_i, k, :,:] = poly_basis @ poly_weights[vtx_idx] + self.V[vtx_idx,:]

                
            
            
            
        output = torch.einsum('fsp,fpsc->fsc', torch.tensor(self.blend_weights, dtype=torch.float32), torch.tensor(self.vertex_funcs, dtype=torch.float32))
        self.output=output
            
        return output


    def build_output_surface(self):

        base_faces = np.asarray(self.base_facepatch.triangles)
        vertex_offset = 0
        num_samples = self.x.shape[0]
        
        vertices_list = []
        vertices_list_Xb = []
        colors_list = []
        colors_list_eq = []
        faces_list = []
        
        for f_i in range(self.F.shape[0]):

            verts = self.F[f_i, :]  # (3,)
    
            # Coarse triangle vertices (3,3)
            V_face = torch.tensor(self.V[verts, :], dtype=torch.float32)
    
            # --- Blended (your main construction) ---
            W = self.blend_weights[f_i]  # (S,3)
            V_patch = self.output[f_i,:,:]
            V_patch_Xb = self.Xb[f_i,:,:]
            vertices_list.append(V_patch)
            vertices_list_Xb.append(V_patch_Xb)
    
            # --- Colors = blend weights ---
            #colors_list.append(W)
            cmap = plt.cm.hsv

            #print( cmap(self.angles[f_i, :, :]).shape)
            colors_list.append( torch.tensor( np.einsum( 'sp,psc->sc', np.asarray(W),  cmap(self.angles[f_i, :, :] / (2*np.pi))[:,:,:-1])))
            colors_list_eq.append( torch.tensor( np.einsum( 'sp,psc->sc', np.asarray(W),  cmap(self.angles_eq[f_i, :, :] / (2*np.pi))[:,:,:-1])))
    
            # Connectivity
            faces_list.append(base_faces + vertex_offset)
            vertex_offset += num_samples
    
        # Concatenate
        V_all = torch.cat(vertices_list, dim=0).cpu().numpy()
        V_all_Xb = torch.cat(vertices_list_Xb, dim=0).cpu().numpy()
        C_all = torch.cat(colors_list, dim=0).cpu().numpy()
        C_all_eq = torch.cat(colors_list_eq, dim=0).cpu().numpy()
        F_all = np.vstack(faces_list)
    
        # --- Blended mesh ---
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(V_all)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(F_all)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(C_all)
        mesh_o3d.compute_vertex_normals()

        mesh_o3d_Xb = copy.deepcopy(mesh_o3d)
        mesh_o3d_Xb.vertices = o3d.utility.Vector3dVector(V_all_Xb)
        mesh_o3d_Xb.compute_vertex_normals()

        mesh_o3d_eq = copy.deepcopy(mesh_o3d_Xb)
        mesh_o3d_eq.vertices = o3d.utility.Vector3dVector(V_all)
        mesh_o3d_eq.vertex_colors = o3d.utility.Vector3dVector(C_all_eq)
        mesh_o3d_eq.compute_vertex_normals()

        return mesh_o3d, mesh_o3d_Xb, mesh_o3d_eq


    
    def onering_coords_func(self, x, i, j, vtx_idx, verts):

        cumulative_angles_a = self.onerings[vtx_idx]['cumulative_angles'] 
        valence = self.onerings[vtx_idx]['valence']  


        
        bary = torch.stack([self.bary_weight(x, i) for i in range(3)]).transpose(1, 0)
        Xb = torch.tensor(bary, dtype = torch.float32) @ verts

        #print(Xb.shape, verts[0].shape)
        a,b,c = verts[(i + 0) % 3].to(x.device), verts[(i + 1) % 3].to(x.device), verts[(i + 2) % 3].to(x.device)

        
        #3 displacement vectors to the vertices
        v_a = Xb - a
        v_b = Xb - b
        v_c = Xb - c


        r_a = torch.sqrt((v_a ** 2).sum(-1))
        r_b = torch.sqrt((v_b ** 2).sum(-1))
        r_c = torch.sqrt((v_c ** 2).sum(-1))
    
        v = v_a
        r = r_a


        theta = torch.atan2(v[:, 1], v[:, 0])
        alpha = torch.atan2((b-a)[1], (b-a)[0])
        theta -= alpha
        
        
        theta = theta + (theta < 0) * two_pi
        theta = theta - (theta >= two_pi) * two_pi

        max_angle = torch.atan2((c-a)[1], (c-a)[0]) - torch.atan2((b-a)[1], (b-a)[0])
        #max_angle = cumulative_angles_a[(j-2)%valence] - cumulative_angles_a[(j-3)%valence]
        max_angle = max_angle + (max_angle < 0) * two_pi
        max_angle = max_angle - (max_angle >= two_pi) * two_pi
        #max_angle = max_angle + (two_pi - 2*max_angle)*(max_angle>(two_pi/2.0))
        
        theta = max_angle - theta

        #print('val', len(cumulative_angles_a), valence)
        #print(np.asarray(cumulative_angles_a) * 180 / np.pi, max_angle*180/np.pi)

        global_theta = ((theta + cumulative_angles_a[(j-2)%valence]) *    ( two_pi / cumulative_angles_a[-1]  ) ) 
        new_theta = global_theta

        new_theta = new_theta + (new_theta < 0) * two_pi
        new_theta = new_theta - (new_theta >= two_pi) * two_pi

        onering_x = torch.stack([r * torch.cos(new_theta), r * torch.sin(new_theta)],
                                dim=0).to(dtype=torch.float32, device=self.device).squeeze().transpose(-1,0)
        




        #### comparison

        # Ensure base_triangle_verts is on the same device
        v_a = x - self.base_triangle_verts[i].to(x.device)
        v_b = x - self.base_triangle_verts[(i + 1) % 3].to(x.device)
        v_c = x - self.base_triangle_verts[(i + 2) % 3].to(x.device)


        r_a = torch.sqrt((v_a ** 2).sum(-1))
        r_b = torch.sqrt((v_b ** 2).sum(-1))
        r_c = torch.sqrt((v_c ** 2).sum(-1))
    
        v = v_a
        r = r_a

    
        theta = torch.atan2(v[:, 1], v[:, 0])
        
        theta -= i * (two_pi / 3)
        
        
        theta = theta + (theta < 0) * two_pi
        theta = theta - (theta >= two_pi) * two_pi
        theta = two_pi / 6 - theta

        theta=theta.clamp(0.0, two_pi/6)


        global_theta = ((theta + (j - 1) * two_pi / 6) * (6 / valence))
        new_theta_eq = global_theta
            
    
        new_theta_eq = new_theta_eq + (new_theta_eq < 0) * two_pi
        new_theta_eq = new_theta_eq - (new_theta_eq >= two_pi) * two_pi


        onering_x_eq = torch.stack([r * torch.cos(new_theta_eq), r * torch.sin(new_theta_eq)],
                                dim=0).to(dtype=torch.float32, device=self.device).squeeze().transpose(-1,0)

        self.onering_x = onering_x
        self.onering_x_eq = onering_x_eq

        return onering_x_eq, (r_a, r_b, r_c), theta, new_theta, new_theta_eq 

        
    def build_blended_mesh(self):
        """
        Builds:
        - blended mesh (your method)
        - barycentric reference mesh (pure affine interpolation)
        - vertex colors = blend weights
    
        Returns:
            mesh_o3d, mesh_tm, bary_mesh_o3d, bary_mesh_tm
        """
    
        blend_weights = self.compute_blend_weights()  # (F, S, 3)


        
    
        # --- Canonical barycentric coordinates (shared across faces) ---
        x = self.x
        bary = torch.stack(
            [self.bary_weight(x, i) for i in range(3)]
        ).transpose(1, 0)  # (S,3)
    
        dtype = blend_weights.dtype
        device = blend_weights.device
    
        base_faces = np.asarray(self.base_facepatch.triangles)
        num_samples = x.shape[0]
        num_faces = self.F.shape[0]
    
        vertices_list = []
        bary_vertices_list = []
        faces_list = []
        colors_list = []
    
        vertex_offset = 0
    
        for face_index in range(num_faces):
            verts = self.F[face_index, :]  # (3,)
    
            # Coarse triangle vertices (3,3)
            V_face = torch.tensor(self.V[verts, :], dtype=dtype, device=device)
    
            # --- Blended (your main construction) ---
            W = blend_weights[face_index]  # (S,3)
            V_patch = W @ V_face
            vertices_list.append(V_patch)
    
            # --- TRUE barycentric reference (your correct version) ---
            V_bary_patch = bary.to(device) @ V_face
            bary_vertices_list.append(V_bary_patch)
    
            # --- Colors = blend weights ---
            colors_list.append(W)
    
            # Connectivity
            faces_list.append(base_faces + vertex_offset)
            vertex_offset += num_samples
    
        # Concatenate
        V_all = torch.cat(vertices_list, dim=0).cpu().numpy()
        V_bary = torch.cat(bary_vertices_list, dim=0).cpu().numpy()
        C_all = torch.cat(colors_list, dim=0).cpu().numpy()
        F_all = np.vstack(faces_list)
    
        # --- Blended mesh ---
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(V_all)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(F_all)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(C_all)
        mesh_o3d.compute_vertex_normals()
    
        mesh_tm = trimesh.Trimesh(
            vertices=V_all,
            faces=F_all,
            vertex_colors=C_all,
            process=False
        )
    
        # --- Barycentric reference mesh ---
        bary_mesh_o3d = o3d.geometry.TriangleMesh()
        bary_mesh_o3d.vertices = o3d.utility.Vector3dVector(V_bary)
        bary_mesh_o3d.triangles = o3d.utility.Vector3iVector(F_all)
        bary_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(C_all)
        bary_mesh_o3d.compute_vertex_normals()
    
        bary_mesh_tm = trimesh.Trimesh(
            vertices=V_bary,
            faces=F_all,
            vertex_colors=C_all,
            process=False
        )
    
        return mesh_o3d, mesh_tm, bary_mesh_o3d, bary_mesh_tm


    

 



        