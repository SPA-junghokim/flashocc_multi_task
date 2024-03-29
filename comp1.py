class Virtual_DepthNet(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 camera_channels=10,
                 with_cp=False,
                 *args, **kwargs):
        super(Virtual_DepthNet, self).__init__()
        self.with_cp = with_cp
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # 生成context feature
        self.context_conv = nn.Sequential(
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128
            )),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )

        self.bn = nn.BatchNorm1d(camera_channels)
        self.context_mlp = Mlp(camera_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))     # (B*N_views, 27)
        x = self.reduce_conv(x)     # (B*N_views, C_mid, fH, fW)
        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)    # (B*N_views, C_mid, fH, fW)
        context = self.context_conv(context)        # (B*N_views, C_context, fH, fW)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, x)
        else:
            depth = self.depth_conv(x)
        return torch.cat([depth, context], dim=1)