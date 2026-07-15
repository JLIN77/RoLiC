# Cross-Modality Feature Transformers for BEVFusion

This document describes how to integrate the proposed **L2C Transformer**, **C2L Transformer**, and **Sparse Similarity Loss (SSL)** into the original BEVFusion framework.

---

# 1. Overview

The proposed framework introduces two bidirectional feature completion modules:

* **L2C Transformer**

  * Recover Camera BEV features from LiDAR BEV features.

* **C2L Transformer**

  * Recover LiDAR BEV features from Camera BEV features.

* **Sparse Similarity Loss (SSL)**

  * Align recovered local BEV features with teacher features extracted from complete modalities.

The overall pipeline is illustrated below:

```text
Camera Images
      │
      ▼
 Camera Encoder
      │
      ▼
 Camera BEV Feature (BEVc')
      │
      │
      ├────────────┐
      │            │
      ▼            ▼
    L2C          C2L
      │            │
      ▼            ▼
 Enhanced      Enhanced
 Camera BEV    LiDAR BEV
      │            │
      └─────Fusion─┘
             │
             ▼
      Detection Head
```

---

# 2. Integration Location

In the original BEVFusion implementation:

```text
mmdet3d/
└── models/
    ├── fusion_models/
    │   └── bevfusion.py
    ├── vtransforms/
    ├── backbones/
    └── heads/
```

The proposed module should be inserted **after BEV feature extraction and before feature fusion**.

Original pipeline:

```python
camera_bev = camera_encoder(...)
lidar_bev  = lidar_encoder(...)

fused_bev = fuse(camera_bev, lidar_bev)
```

Modified pipeline:

```python
camera_bev = camera_encoder(...)
lidar_bev  = lidar_encoder(...)

camera_bev = L2C(
    camera_bev,
    lidar_bev
)

lidar_bev = C2L(
    camera_bev,
    lidar_bev
)

fused_bev = fuse(
    camera_bev,
    lidar_bev
)
```

---

# 3. File Structure

Create a new directory:

```text
mmdet3d/models/cross_modal/
│
├── l2c_transformer.py
├── c2l_transformer.py
├── cross_attention.py
├── local_patch.py
├── ssl_loss.py
└── __init__.py
```

Example:

```python
from .l2c_transformer import L2CTransformer
from .c2l_transformer import C2LTransformer
from .ssl_loss import SparseSimilarityLoss
```

