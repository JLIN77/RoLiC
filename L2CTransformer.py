class L2CTransformer(nn.Module):

    def __init__(self, c_dim):

        super().__init__()

        self.extract = LocalPatchExtraction()

        self.dim_reduce = nn.Conv2d(
            c_dim,
            c_dim,
            kernel_size=1
        )

        self.norm = nn.BatchNorm2d(c_dim)

        self.attn = CrossAttention(c_dim)

    def forward(
        self,
        bev_c_incomplete,
        bev_l_complete,
        mask_c
    ):

        # BEVc'_loc
        bev_c_loc = self.extract(
            bev_c_incomplete,
            mask_c
        )

        # BEVl_loc
        bev_l_loc = self.extract(
            bev_l_complete,
            mask_c
        )

        bev_l_loc = self.dim_reduce(
            bev_l_loc
        )

        fused = self.norm(
            bev_c_loc + bev_l_loc
        )

        bev_c_loc_l2c, attn = self.attn(
            q_feat=bev_l_loc,
            kv_feat=fused
        )

        bev_c = (
            bev_c_incomplete * (1 - mask_c)
            + bev_c_loc_l2c * mask_c
        )

        return bev_c, bev_c_loc_l2c, attn