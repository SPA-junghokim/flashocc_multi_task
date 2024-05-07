def forward(self, input, stereo_metas=None):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
                mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape

        x_input = x.view(B * N, C, H, W)      # (B*N_views, C, fH, fW)
        x_depth, middle_feat = self.depth_net(x_input, mlp_input)      # (B*N_views, D+C_context, fH, fW)
        
        depth_digit = x_depth[:, :self.depth_channels, ...]    # (B*N_views, D, fH, fW)
        self.depth_feat = depth_digit # for focal loss

        if self.LSS_Rendervalue:
            if self.depth_render_sigmoid:
                self.transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                self.rendering_value = self.transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))
            else:
                self.transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                self.rendering_value = self.transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))
        

        tran_feat = x_depth[:, self.depth_channels:self.depth_channels + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)
            
        depth = depth_digit.softmax(dim=1)  # (B*N_views, D, fH, fW)
        img_feat_with_depth = depth.unsqueeze(1) * tran_feat.unsqueeze(2)
        
        ida = torch.zeros(B, N, 4, 4)
        ida[:, :, :3, :3] = post_rots
        ida[:, :, :3, 3] = post_trans
        ida[:, :, 3, 3] = 1
        

        geom_xyz, geom_xyz_valid = self.get_geometry_collapsed(
            rots,
            trans,
            intrins,
            post_rots, 
            post_trans,
            bda,
            z_min=-1.,
            z_max=5.4)
        
        geom_xyz_valid = self._split_batch_cam(geom_xyz_valid, inv=True).unsqueeze(1)
        img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3)
        img_context = img_feat_with_depth
        img_context = self._split_batch_cam(img_context)
        img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous()
        
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz[..., 2] = 0  # collapse z-axis
        geo_pos = torch.ones_like(geom_xyz)
        
        feature_map, _ = average_voxel_pooling(geom_xyz, img_context, geo_pos, self.voxel_num.cuda())
        
        # bev_feat, depth = self.view_transform(input, depth, tran_feat, kept)
        
        return feature_map.contiguous(), depth_digit.softmax(dim=1), middle_feat