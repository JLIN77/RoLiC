class LocalPatchExtraction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feat, mask):

        # 提取缺失区域

        return feat * mask