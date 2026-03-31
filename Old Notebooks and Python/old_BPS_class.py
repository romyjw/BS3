

class BPS(nn.Module):
    def __init__(self, layer_sizes, posenc = None, he_mesh=None, from_file = False, coarse_patches_id=None, device=None, overlap_param=0.6, degree=3, global_scale=1.0,
                 blend_type = 'radial', virtual_vertex_flag=True, angle_flag = 'equal'):
        super(BPS, self).__init__()

        self.device = device
        self.angle_flag = angle_flag

        if from_file==False:
            self.he_mesh=he_mesh
            self.patches_path = None
        else:
            #self.patches_path = 'patches/'+coarse_patches_id
            #filepath = self.patches_path+'/coarse.obj'
            filepath = 'data/surfaces/'+coarse_patches_id+'.obj'
            coarse_mesh_o3d = o3d.io.read_triangle_mesh(filepath)
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

        #precise control over the blend function
        self.overlap_param = overlap_param
        self.posenc = posenc
        self.blend_type = blend_type
        self.global_scale = global_scale
        self.degree = degree

        self.disp_modifier = None
        
        #self.virtual_vertex_flag = virtual_vertex_flag
        
        

        # a halfedge is assigned to each vertex (onering)
        print('Assiging a halfedge to each vertex.')
        initial_V_he = get_V_he(self.he_mesh)
        
        self.V_he = optimise_V_he(self.he_mesh, initial_V_he)

        self.base_triangle_verts = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, torch.sqrt(torch.tensor(3.0, dtype=torch.float32, device=self.device)) / 2],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        print('base triangle verts', self.base_triangle_verts.dtype)


        print('Computing onering data.')

        self.onerings = compute_onering_data(self.he_mesh, self.V_he, vtx_normals=self.vertex_normals)
        
        self.mlps = nn.ModuleList([ResidualMLP(layer_sizes, init_type='kaiming') for _ in range(self.V.shape[0])])
        self.id_mlps = nn.ModuleList([ResidualMLP(layer_sizes, init_type='zero') for _ in range(self.V.shape[0])])

        
        self.set_poly_coeffs_to_default(degree=self.degree)
        

        print('Computing rotations.')
        self.compute_fixed_rotations()

        self.default_coarse_points = self.V
        self.default_rotations = self.rotations


        self.test_scale = 1.0


    

    def set_poly_coeffs_to_default(self, degree=3):
        # number of coefficients in 2D polynomial basis of given degree
        size = int((degree + 1) * (degree + 2) / 2)
    
        self.poly_coeffs = nn.ModuleList()
    
        for _ in range(self.V.shape[0]):  # loop over vertices
            coeff_list = []
            for k in range(size):
                vec = torch.zeros(3, device=self.device)
    
                # preserve your old defaults:
                if k == 1:   # first derivative wrt u
                    vec[0] = 1.0
                elif k == 2: # first derivative wrt v
                    vec[1] = 1.0
    
                coeff_list.append(nn.Parameter(vec))
    
            self.poly_coeffs.append(nn.ParameterList(coeff_list))



    def get_poly_matrix(self, vtx_idx, degree=None):
        """Return coefficient matrix (num_terms × 3) up to `degree`."""
        if degree is None:
            degree = self.degree
        num_terms = int((degree + 1) * (degree + 2) / 2)
        coeffs = torch.stack([p for p in self.poly_coeffs[vtx_idx][:num_terms]], dim=0)
        return coeffs  # shape: (num_terms, 3)

  

        

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
    




    
        
    def reset(self):
        self.rotations = self.default_rotations
        self.V = self.default_coarse_points
        self.scaling_weights = [1 for i in range(self.n)]

    def compute_fixed_rotations(self):
        self.rotations = []
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
        v_a = x - self.base_triangle_verts[i]
        v_b = x - self.base_triangle_verts[(i + 1) % 3]
        v_c = x - self.base_triangle_verts[(i + 2) % 3]
    
        # Compute 2D cross product as scalar
        cross = v_b[:,0] * v_c[:,1] - v_b[:,1] * v_c[:,0]
    
        bary_weight = 0.5 * abs(cross) / (0.5 * 0.5 * np.sqrt(3))
    
        return bary_weight

    def discrete_weight(self, x, i):
        v_a = x - self.base_triangle_verts[i]
        side_vec1 = self.base_triangle_verts[(i+1)%3] - self.base_triangle_verts[i]
        side_vec2 = self.base_triangle_verts[(i+2)%3] - self.base_triangle_verts[i]

        dot1 = (v_a*side_vec1).sum(-1)
        dot2 = (v_a*side_vec2).sum(-1)

        discrete_weight = torch.logical_and(dot1 <= 0.5, dot2 <= 0.5).float()

        return discrete_weight
        



    def onering_coords(self, x, i, j, angles, flag='spline'):
        #device = x.device  # ensure all tensors use the same device
        #print('oc device', device)
    
        valence = len(angles)
    
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

        theta.clamp(0.0, two_pi/6)

        
        global_theta = ((theta + (j - 1) * two_pi / 6) * (6 / valence))
    
        
        
        if flag == 'equal':
            new_theta = global_theta

            
        elif flag in ['linear', 'spline']:
            frac = theta / (two_pi / 6)
            angles_tensor = torch.tensor(angles, dtype=torch.float32, device=self.device)
    
            normalised_angles = two_pi * angles_tensor / angles_tensor.sum()
            cumulative_angles = torch.cat([
                torch.tensor([0.0], dtype=torch.float32, device=self.device),
                normalised_angles.cumsum(0),
                torch.tensor([two_pi + normalised_angles[0]], dtype=torch.float32, device=self.device)
            ])
    
            if flag == 'linear':
                start_angle = cumulative_angles[j]
                end_angle = cumulative_angles[j + 1]
                angle_diff = end_angle - start_angle
                new_theta = start_angle + frac * angle_diff - cumulative_angles[1]
    
            elif flag == 'spline':
                ctrl_pts = torch.cat([
                    cumulative_angles[:-1] - two_pi,
                    cumulative_angles[:-1],
                    cumulative_angles[:-1] + two_pi
                ], dim=0)
    
                x_vals = torch.tensor([k - valence for k in range(valence * 3)], dtype=torch.float32, device=self.device) * two_pi / valence
    
                # Spline coefficients and evaluation
                coeffs = compute_spline_coeffs(x_vals, ctrl_pts)
                new_theta = cubic_spline_eval(x_vals, coeffs, global_theta)
        else:
            raise Exception(f"{flag} is not a valid flag")
    
        new_theta = new_theta + (new_theta < 0) * two_pi
        new_theta = new_theta - (new_theta >= two_pi) * two_pi
    
        onering_x = torch.stack([r * torch.cos(new_theta), r * torch.sin(new_theta)],
                                dim=0).to(dtype=torch.float32, device=self.device).transpose(1, 0)
    
        return onering_x, (r_a, r_b, r_c), theta, new_theta




    def blend_weight(self, x, radii, i=0, j=0, cur_patch_index = 0, blend_type=None):

        r_a, r_b, r_c = radii
                
        if blend_type=='bary':
            bary_weight = self.bary_weight(x[cur_patch_index,:,:].squeeze(), i)
            blend_weight = bary_weight
            
        elif blend_type=='radial_1':
            blend_weight = B1(r_a, v=self.overlap_param)

        elif blend_type=='pou_1':
            blend_weight = B1(r_a, v=self.overlap_param) / ( B1(r_a, v=self.overlap_param) + B1(r_b, v=self.overlap_param) + B1(r_c, v=self.overlap_param)     )

        elif blend_type=='radial_2':
            blend_weight = B2(r_a, v=self.overlap_param)

        elif blend_type=='pou_2':
            blend_weight = B2(r_a, v=self.overlap_param) / ( B2(r_a, v=self.overlap_param) + B2(r_b, v=self.overlap_param) + B2(r_c, v=self.overlap_param)     )
            
        elif blend_type=='discrete':
            blend_weight = self.discrete_weight(x[cur_patch_index,:,:].squeeze(), i)

        else:
            raise Exception('Sorry, did not understand this blend type:', blend_type)

        return blend_weight


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
        torch.save([vtx_poly_coeffs for vtx_poly_coeffs in self.poly_coeffs], filename)

    def load_poly_coeffs(self, filename="poly_coeffs.pth"):
        poly_coeffs_data = torch.load(filename, map_location=self.device)
        self.poly_coeffs = nn.ParameterList([nn.Parameter(c) for c in poly_coeffs_data])
    '''


    
    def save_mlps(self, filename="mlps.pth"):
        torch.save({
            'mlps_state_dict': [mlp.state_dict() for mlp in self.mlps],
        }, filename)
    
    def load_mlps(self, filename="mlps.pth"):
        checkpoint = torch.load(filename)
        
        for mlp, state_dict in zip(self.mlps, checkpoint['mlps_state_dict']):
            mlp.load_state_dict(state_dict)



    def actual_vertex_positions(self):
        actual_vertex_positions = []
        for i in range(self.V.shape[0]):

            offset = self.get_poly_matrix(i)[0,:]
            cur_vtx_position =  offset @ self.rotations[i] + self.V[i]
            #cur_vtx_position = self.V[i]

            actual_vertex_positions.append(cur_vtx_position)
            
        return actual_vertex_positions


        
    
    def forward(self, x, test_flag = None, return_extra_data=False, return_onering_coords=False, select_patch_indices=None,
               degree = None):


        
        if degree is None:
            degree = self.degree


            
        if test_flag == 'unblended_quad':
            output = torch.zeros((self.F.shape[0], self.V.shape[0], x.shape[1], 3), device=self.device)
        else:
            output = torch.zeros((self.F.shape[0], x.shape[1], 3), device=self.device)
        
        if return_extra_data:
            extra_data = {'blend': np.zeros( (  self.F.shape[0], self.V.shape[0], output.shape[1] )  ), #patch*vert*samples
                    'angle': -10*np.ones( (self.F.shape[0], self.V.shape[0], output.shape[1]) ), #patch*vert*samples
                    'r': 10000 * np.ones( (self.F.shape[0], self.V.shape[0], output.shape[1]) ), #patch*vert*samples
                'per_vertex_unblended': np.zeros( (  self.F.shape[0], self.V.shape[0], output.shape[1], 3 )  )  } #patch*vert*samples*3

        if select_patch_indices is None:
            select_patch_indices = range(self.F.shape[0])


        
        for patch_index in select_patch_indices:
            verts = self.F[patch_index,:]


            #virtual_vertex = ( self.V[verts[0]] + self.V[verts[1]] + self.V[verts[2]] )/3
            
            #blendsum = torch.zeros((x.shape[1],1))
            
            for i in range(3):
            
                onering = self.onerings[verts[i]]
                #valence = onering['valence']
                #which face in onering matrix?
                
                j = onering['triangles'].index(patch_index)
                #onering_x, radii, theta, new_theta = self.onering_coords_equal_angles(x[patch_index,:,:].squeeze(), i, j, onering['angle'], onering['angle_stretch'])
                onering_x, radii, theta, new_theta = self.onering_coords(x[patch_index,:,:].squeeze(), i, j, onering['tp_angles'], flag=self.angle_flag)

                blend_weight = self.blend_weight(x, radii, i, j, blend_type=self.blend_type)
                r_a, r_b, r_c = radii

                if return_extra_data:

                    extra_data['blend'][patch_index, verts[i], :] += blend_weight.detach().cpu().numpy()
                    
                    extra_data['angle'][patch_index, verts[i], :] = new_theta.detach().cpu()
                    extra_data['r'][patch_index, verts[i], :] = r_a.detach().cpu()


                
                
                if test_flag == 'zero':
                    output[patch_index, :, :] +=  blend_weight.unsqueeze(-1)*( self.V[verts[i]] )
                    
                elif test_flag == 'id':
                    output[patch_index, :, :] +=  blend_weight.unsqueeze(-1)*( self.global_scale * self.id_mlps[verts[i]](self.posenc(onering_x))@self.rotations[verts[i]] + self.V[verts[i]] )

                    if return_extra_data:
                        extra_data['per_vertex_unblended'][patch_index, verts[i], :, :] = (self.global_scale * self.id_mlps[verts[i]](self.posenc(onering_x))@self.rotations[verts[i]] + self.V[verts[i]]).detach().cpu().numpy()
                

                elif test_flag == 'poly' or test_flag is None:
                    
                    XY = self.global_scale * onering_x
                    X = XY[:, 0]
                    Y = XY[:, 1]
                    
                    basis = self.poly_basis(X, Y, degree).to(x.device)
                    A = self.get_poly_matrix(verts[i], degree).to(x.device)

                    
                    PXY = basis @ A  # (num_samples × 3)

                    output[ patch_index, :, :] += blend_weight.unsqueeze(-1) * (  PXY   @self.rotations[verts[i]].to(x.device) + self.V[verts[i]].to(x.device) )
                    if return_extra_data:
                        extra_data['per_vertex_unblended'][patch_index, verts[i], :, :] = (PXY   @self.rotations[verts[i]].to(x.device) + self.V[verts[i]].to(x.device)).detach().cpu().numpy()


        if self.disp_modifier is not None:
            bs_normals = diffmod.compute_normals(out=output, wrt=x)
            #bs_normals = torch.nan_to_num(bs_normals, nan=0.0) #ugly, do not leave in



            
            print('using disp modifier')
            old=output
            output = self.disp_modifier(output, bs_normals)

            print('are they different:', ((old-output)**2).mean())
                
                


        if return_extra_data:    
            return output, extra_data

        if return_onering_coords:
            return output, onering_x

        return output




