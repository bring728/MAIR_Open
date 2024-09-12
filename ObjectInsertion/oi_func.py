from oi_utils import *
# from Zhu.render.brdf import *
# from Zhu.zhu_utils import create_frame


def mair_object_rendering(mesh, data, rays, gpu, flags, self_occ, object_type):
    size = flags.size
    eps = 0.001
    all_pts, all_rays, index_tri = mesh.ray.intersects_location(rays['origin'], rays['ray'],
                                                                multiple_hits=False)
    if all_pts.shape[0] == 0:
        return
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[index_tri], points=all_pts)
    all_normal = trimesh.unitize(
        (mesh.vertex_normals[mesh.faces[index_tri]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

    if object_type == 'chrome':
        chunk = 50000
        mesh_points_rub = torch.from_numpy(all_pts).to(gpu).float()
        mesh_points_rdf = mesh_points_rub * data['rub_to_rdf'][None]
        mesh_viewdir_rdf = rays['viewdir'][all_rays] * data['rub_to_rdf'][None]
        # mesh_normal_rub = -torch.from_numpy(mesh.face_normals[index_tri]).to(gpu).float()
        mesh_normal_rub = torch.from_numpy(all_normal).to(gpu).float()
        mesh_normal_rdf = (mesh_normal_rub * data['rub_to_rdf'][None])

        mesh_reflection_rdf = -mesh_viewdir_rdf + torch.sum(mesh_normal_rdf * mesh_viewdir_rdf, dim=-1,
                                                            keepdim=True) * mesh_normal_rdf * 2.0
        mesh_reflection_rdf = mesh_reflection_rdf.unsqueeze(-2)
        mesh_envmaps = []
        for i in range(0, mesh_points_rdf.shape[0], chunk):
            mesh_envmap = envmapfromVSG(data['VSG'], mesh_points_rdf[None, None, i:i + chunk, None, :],
                                        mesh_reflection_rdf[None, None, i:i + chunk, :, :],
                                        data['r_dist'], data['bb'], flags.sg_order)
            mesh_envmaps.append(mesh_envmap)
        mesh_envmaps = torch.cat(mesh_envmaps, dim=2)
        mesh_envmaps = (mesh_envmaps * flags.env_scale * data['light_scale']).squeeze()
        radiance = torch.clamp(mesh_envmaps, 0, 1.0)

    else:
        chunk = 3000
        diff_list = []
        spec_list = []
        for i in range(0, all_pts.shape[0], chunk):
            mesh_viewdir_rub = rays['viewdir'][all_rays[i:i + chunk]]
            mesh_normal_rub = torch.from_numpy(all_normal[i:i + chunk]).to(gpu).float()
            mesh_points_rub = torch.from_numpy(all_pts[i:i + chunk]).to(gpu).float()
            mesh_points_rdf = (mesh_points_rub * data['rub_to_rdf'][None])[None, None, :, None, :]
            num_pts = mesh_points_rub.shape[0]

            if flags.sampling.type == 'importance' and object_type != 'shadow':
                cx, cy, cz = create_frame(rearrange(mesh_normal_rub, 'w c -> () c () w'))
                wi_world = rearrange(mesh_viewdir_rub, 'w c -> () c () w')
                wi_x = torch.sum(cx * wi_world, dim=1)
                wi_y = torch.sum(cy * wi_world, dim=1)
                wi_z = torch.sum(cz * wi_world, dim=1)
                wi = torch.stack([wi_x, wi_y, wi_z], dim=1)
                # wi_mask = torch.where(wi[:, 2:3, ...] < 0.001, torch.zeros_like(wi[:, 2:3, ...]),
                #                       torch.ones_like(wi[:, 2:3, ...]))

                wi[:, 2] = torch.clamp(wi[:, 2], min=1e-3)
                wi = F.normalize(wi, dim=1, eps=1e-6)
                wi = wi.unsqueeze(1)  # (bn, 1, 3, h, w)

                albedo_clip = data['albedo'][None, :, None, None]
                rough_clip = torch.tensor([data['rough']], dtype=albedo_clip.dtype).to(gpu)[None, :, None, None]
                metal_clip = torch.tensor([0.0], dtype=albedo_clip.dtype).to(gpu)[None, :, None, None]

                samples = torch.rand(1, flags.sampling.spp, 3, 1, num_pts).to(gpu)
                albedo_clip = albedo_clip.repeat(1, 1, 1, num_pts)
                rough_clip = rough_clip.repeat(1, 1, 1, num_pts)
                metal_clip = metal_clip.repeat(1, 1, 1, num_pts)

                wo_diffuse = square_to_cosine_hemisphere(samples[:, :, 1:, ...])
                if flags.sampling.brdf_type == "ggx":
                    specularF0 = (baseColorToSpecularF0(albedo_clip, metal_clip))
                    diffuseReflectance = (albedo_clip * (1 - metal_clip))
                    kS = probabilityToSampleSpecular(diffuseReflectance, specularF0)
                    sample_diffuse = samples[:, :, 0, ...] >= kS
                    wo_specular = sample_ggx_specular(samples[:, :, 1:, ...], rough_clip, wi)
                else:
                    diffuseRatio = 0.5 * (1.0 - metal_clip)
                    sample_diffuse = samples[:, :, 0, ...] < diffuseRatio  # (bn, spp, h, w)
                    wo_specular = sample_disney_specular(samples[:, :, 1:, ...], rough_clip, wi)
                wo = torch.where(sample_diffuse.unsqueeze(2).expand(1, flags.sampling.spp, 3, 1, num_pts), wo_diffuse,
                                 wo_specular)  # (bn, spp, 3, h, w)

                if flags.sampling.brdf_type == "ggx":
                    pdfs = pdf_ggx(albedo_clip, rough_clip, metal_clip, wi, wo).unsqueeze(2)
                    eval_diff, eval_spec, mask = eval_ggx(albedo_clip, rough_clip, metal_clip, wi, wo)
                else:
                    pdfs = pdf_disney(rough_clip, metal_clip, wi, wo).unsqueeze(2)
                    eval_diff, eval_spec, mask = eval_disney(albedo_clip, rough_clip, metal_clip, wi, wo)
                mesh_ls_rub = (wo[:, :, 0:1, ...] * cx.unsqueeze(1) + wo[:, :, 1:2, ...] *
                               cy.unsqueeze(1) + wo[:, :, 2:3, ...] * cz.unsqueeze(1))
                mesh_ls_rdf = rearrange(mesh_ls_rub, 'b l c h w -> b h w l c') * data['rub_to_rdf']

                pdfs = torch.clamp(pdfs, min=0.001)
                ndl = torch.clamp(wo[:, :, 2:, ...], min=0)
                brdfDiffuse = eval_diff.expand([1, flags.sampling.spp, 3, 1, num_pts]) * ndl / pdfs
                brdfDiffuse = rearrange(brdfDiffuse, 'b l c h w -> b h w l c')
                brdfSpec = eval_spec.expand([1, flags.sampling.spp, 3, 1, num_pts]) * ndl / pdfs
                brdfSpec = rearrange(brdfSpec, 'b l c h w -> b h w l c')

            else:
                N2C = get_N2C(mesh_normal_rub, data['up'])
                mesh_ls_rub = (N2C.unsqueeze(-3) @ data['ls'][..., None]).squeeze(-1)
                mesh_ls_rdf = (mesh_ls_rub * data['rub_to_rdf'])[None, None]

            mesh_envmaps = envmapfromVSG(data['VSG'], mesh_points_rdf,
                                         mesh_ls_rdf, data['r_dist'], data['bb'], flags.sg_order)
            mesh_envmaps = mesh_envmaps * flags.env_scale * data['light_scale']
            # mesh_envmaps = rearrange(mesh_envmaps, 'b h w (q p) c -> b c h w q p', q=size[3], p=size[2])

            # for self-occlusion
            if self_occ:
                points_mesh_rub_np = mesh_points_rub[:, None].repeat(1, mesh_ls_rub.shape[1], 1).reshape(
                    [-1, 3]).cpu().numpy()
                ls_mesh_rub_np = mesh_ls_rub.reshape([-1, 3]).cpu().numpy()
                points_mesh_rub_np = points_mesh_rub_np + ls_mesh_rub_np * eps
                _, self_occluded_ray_index, _ = mesh.ray.intersects_location(points_mesh_rub_np, ls_mesh_rub_np,
                                                                             multiple_hits=False)
                mesh_envmaps = mesh_envmaps.reshape([-1, 3])
                mesh_envmaps[self_occluded_ray_index] = 0
                mesh_envmaps = rearrange(mesh_envmaps, '(b h w l) c -> b h w l c', b=1, h=1, w=num_pts)

            if flags.sampling.type == 'importance' and object_type != 'shadow':
                diff = torch.mean(brdfDiffuse * mesh_envmaps, dim=-2)
                spec = torch.mean(brdfSpec * mesh_envmaps, dim=-2)
            else:
                mesh_albedo = torch.ones_like(mesh_normal_rub) * data['albedo']
                mesh_rough = torch.ones_like(mesh_normal_rub)[:, :1] * data['rough']
                diff, spec, _ = pbr(mesh_viewdir_rub[None, None, :, None],
                                    mesh_ls_rub[None, None],
                                    mesh_normal_rub[None, None, :, None],
                                    mesh_albedo[None, None, :, None],
                                    mesh_rough[None, None, :, None],
                                    data['ndotl'], data['envWeight_ndotl'], mesh_envmaps)

            diff_list.append(diff.reshape([-1, 3]))
            spec_list.append(spec.reshape([-1, 3]))

        colorDiff = torch.cat(diff_list, dim=0)
        colorSpec = torch.cat(spec_list, dim=0)

        if flags.saturation:
            radiance = colorDiff + colorSpec
            tau = 0.7
            mask = (radiance > tau).float()
            radiance_saturated = 1 - (1 - tau) * torch.exp(-(radiance - tau) / (1 - tau))
            radiance = radiance * (1 - mask) + radiance_saturated * mask
        else:
            radiance = torch.clamp(colorDiff + colorSpec, 0, 1.0)
    return all_rays, radiance


def zhu_object_rendering(mesh, img, data, cam, gpu, size_fg, object_type):
    mesh_points_rub, index_ray, index_tri = mesh.ray.intersects_location(cam['origin'], cam['ray'],
                                                                         multiple_hits=False)
    if mesh_points_rub.shape[0] == 0:
        return
    # https://github.com/mikedh/trimesh/issues/1285
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[index_tri], points=mesh_points_rub)
    normal_interp = trimesh.unitize(
        (mesh.vertex_normals[mesh.faces[index_tri]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

    mesh_points_rub = torch.from_numpy(mesh_points_rub).to(gpu).float()
    mesh_normal_rub = torch.from_numpy(normal_interp).to(gpu).float()

    pixel_ray = data['pixels'][index_ray]
    mask = np.zeros([size_fg[1], size_fg[0]], dtype=np.float32)
    mask[pixel_ray[:, 1], pixel_ray[:, 0]] = 1.0
    # cx, cy, rad = bounding_box(mask)
    box_all, boxes = bounding_box(mask, chunk=10000)

    data['zhu_n'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = mesh_normal_rub.permute([1, 0])
    data['zhu_d'][pixel_ray[:, 1], pixel_ray[:, 0]] = -mesh_points_rub[:, -1]
    if object_type == 'chrome':
        data['zhu_a'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = 1.0
        data['zhu_r'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = 0.0
        data['zhu_m'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = 1.0
    else:
        data['zhu_a'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = data['albedo'][:, None]
        data['zhu_r'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = data['rough']
        # data['zhu_m'][0, :, pixel_ray[:, 1], pixel_ray[:, 0]] = 0.0

    im = torch.from_numpy(ldr2hdr(img)[None]).float().to(gpu).permute([0, 3, 1, 2])
    vpos = depth_to_vpos(data['zhu_d'], data['fov'], True)
    vpos = vpos.unsqueeze_(0).to(gpu)
    diff_list = []
    spec_list = []
    for box in boxes:
        r_diff, r_spec, _ = data['zhu_renderer'](box, data['zhu_model'],
                                                 im, data['zhu_a'], data['zhu_n'],
                                                 data['zhu_r'], data['zhu_m'], vpos)
        diff_list.append(r_diff)
        spec_list.append(r_spec)
    radiance = torch.cat(diff_list, dim=2) + torch.cat(spec_list, dim=2)
    radiance = torch.where(torch.isfinite(radiance), radiance,
                           torch.zeros_like(radiance))[0]
    radiance = torch.clamp(radiance, 0, 1.0)

    min_cx = max(box_all[0], 0)
    max_cx = min(box_all[1], size_fg[0])
    min_cy = max(box_all[2], 0)
    max_cy = min(box_all[3], size_fg[1])

    im_clip = im[0][:, min_cy:max_cy, min_cx:max_cx]
    mask_clip = mask[min_cy:max_cy, min_cx:max_cx]
    mask_clip = torch.from_numpy(mask_clip[None]).to(gpu)
    im[0, :, min_cy:max_cy, min_cx:max_cx] = im_clip * (1 - mask_clip) + radiance * mask_clip
    return im


def cis_rendering(mesh, val_dict, gpu, env):
    mesh_points_rub, index_ray, index_tri = mesh.ray.intersects_location(val_dict['origin'], val_dict['ray'],
                                                                         multiple_hits=False)

    # https://github.com/mikedh/trimesh/issues/1285
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[index_tri], points=mesh_points_rub)
    normal_interp = trimesh.unitize(
        (mesh.vertex_normals[mesh.faces[index_tri]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

    mesh_viewdir_rub = val_dict['viewdir'][index_ray]
    mesh_normal_rub = torch.from_numpy(normal_interp).to(gpu).float()
    mesh_reflection_rub = -mesh_viewdir_rub + torch.sum(mesh_normal_rub * mesh_viewdir_rub, dim=-1,
                                                        keepdim=True) * mesh_normal_rub * 2.0

    xy = mesh_reflection_rub[:, 0] ** 2 + mesh_reflection_rub[:, 1] ** 2
    pi = torch.atan2(torch.sqrt(xy), mesh_reflection_rub[:, 2])
    theta = torch.atan2(mesh_reflection_rub[:, 1], mesh_reflection_rub[:, 0])

    r = (env.shape[0] * pi / np.pi).int().cpu().numpy()
    c = ((theta + np.pi) / (2 * np.pi) * env.shape[1]).int().cpu().numpy()
    env_mapping = torch.from_numpy(env[r - 1, c - 1])
    env_mapping = torch.clamp(env_mapping, 0, 1.0)
    return index_ray, env_mapping
