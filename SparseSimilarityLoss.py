class SparseSimilarityLoss(nn.Module):

    def __init__(self, tau=0.1):
        super().__init__()

        self.tau = tau

    def forward(
        self,
        pred_feat,
        gt_feat,
        attn
    ):
        """
        pred_feat : B,C,H,W
        gt_feat   : B,C,H,W
        attn      : B,N,N
        """

        B, C, H, W = pred_feat.shape

        # voxel importance

        voxel_score = attn.max(
            dim=-1
        )[0]

        heatmap = (
            voxel_score > self.tau
        ).float()

        pred_feat = pred_feat.flatten(2)
        gt_feat = gt_feat.flatten(2)

        l1 = torch.abs(
            pred_feat - gt_feat
        )

        loss = (
            l1 *
            heatmap.unsqueeze(1)
        ).mean()

        return loss