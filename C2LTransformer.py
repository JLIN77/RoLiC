class C2LTransformer(nn.Module):

    def __init__(
        self,
        cam_dim,
        lidar_dim
    ):
        super().__init__()

        self.extract = LocalPatchExtraction()

        self.dim_expand = nn.Conv2d(
            cam_dim,
            lidar_dim,
            kernel_size=1
        )

        self.norm = nn.BatchNorm2d(
            lidar_dim
        )

        self.attn = CrossAttention(
            lidar_dim
        )

    def forward(
        self,
        bev_c_complete,
        bev_l_incomplete,
        mask_l
    ):

        bev_c_loc = self.extract(
            bev_c_complete,
            mask_l
        )

        bev_l_loc = self.extract(
            bev_l_incomplete,
            mask_l
        )

        bev_c_loc = self.dim_expand(
            bev_c_loc
        )

        fused = self.norm(
            bev_c_loc + bev_l_loc
        )

        bev_l_loc_c2l, attn = self.attn(
            q_feat=bev_c_loc,
            kv_feat=fused
        )

        bev_l = (
            bev_l_incomplete * (1 - mask_l)
            + bev_l_loc_c2l * mask_l
        )

        return bev_l, bev_l_loc_c2l, attn