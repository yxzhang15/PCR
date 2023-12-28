# A Collection of DL-based Dehazing Methods

This repository provides a summary of deep learning based point cloud registration algorithms. 

We classify registration algorithms into supervised and unsupervised, as follows.

[Supervised Point Cloud Registration Methods](#supervised)
 * [1. Descriptor Extraction](#s-1)
   * [1. 1 Keypoint-based](#s-1-1)
   * [1. 2 Keypoint-free](#s-1-2)
   * [1. 3 Multiview](#s-1-3)
 * [2. Feature Enhancement](#s-2)   

[Unsupervised Point Cloud Registration Methods](#unsupervised)

[Dataset](#dataset)

<p id="supervised"></p>

## Supervised Dehazing Methods

<p id="s-1"></p>

### 1. Descriptor Extraction

<p id="s-1-1"></p>

#### 1. 1 Keypoint-based
* DeepVCP: An End-to-End Deep Neural Network for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9009450)\]
\[[code](https://github.com/jundaozhilian/DeepVCP-PyTorch)\]

* The Perfect Match: 3D Point Cloud Matching With Smoothed Densities.
\[[paper](https://ieeexplore.ieee.org/document/8954296)\]
\[[code](https://github.com/zgojcic/3DSmoothNet)\]

* 3D Local Features for Direct Pairwise Registration.
\[[paper](https://ieeexplore.ieee.org/document/8953479)\]
\[[code]()\]

* HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9710151)\]
\[[code](https://ispc-group.github.io/hregnet)\]

* SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9577271)\]
\[[code](https://github.com/QingyongHu/SpinNet)\]

* StickyPillars: Robust and Efficient Feature Matching on Point Clouds using Graph Neural Networks.
\[[paper](https://ieeexplore.ieee.org/document/9578620)\]
\[[code]()\]

* You Only Hypothesize Once: Point Cloud Registration with Rotation-equivariant Descriptors.
\[[paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548023)\]
\[[code](https://github.com/HpWang-whu/YOHO)\]

* RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations.
\[[paper](https://ieeexplore.ieee.org/document/10044259)\]
\[[code](https://github.com/HpWang-whu/RoReg)\]

* BUFFER: Balancing Accuracy, Efficiency, and Generalizability in Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/10205493)\]
\[[code](https://github.com/aosheng1996/BUFFER)\]

<p id="s-1-2"></p>

#### 1. 2 Keypoint-free

* CoFiNet: Reliable Coarse-to-fine Correspondences for Robust PointCloud Registration.
\[[paper](https://proceedings.neurips.cc/paper/2021/hash/c85b2ea9a678e74fdc8bafe5d0707c31-Abstract.html)\]
\[[code](https://github.com/haoyu94/Coarse-to-fine-correspondences)\]

* One-Inlier is First: Towards Efficient Position Encoding for Point Cloud Registration.
\[[paper](https://papers.nips.cc/paper_files/paper/2022/hash/2e163450c1ae3167832971e6da29f38d-Abstract-Conference.html)\]
\[[code]()\]

* GeDi: Learning General and Distinctive 3D Local Deep Descriptors for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9775606)\]
\[[code]()\]

* GeoTransformer: Fast and Robust Point Cloud Registration With Geometric Transformer.
\[[paper](https://ieeexplore.ieee.org/document/10076895)\]
\[[code](https://github.com/qinzheng93/GeoTransformer)\]

* RoITr: Rotation-Invariant Transformer for Point Cloud Matching.
\[[paper](https://ieeexplore.ieee.org/document/10204543)\]
\[[code](https://github.com/haoyu94/RoITr)\]

<p id="s-1-3"></p>

#### 1. 3 Multiview

* Learning and Matching Multi-View Descriptors for Registration of Point Clouds.
\[[paper](https://dl.acm.org/doi/abs/10.1007/978-3-030-01267-0_31)\]
\[[code]()\]

* End-to-end learning local multi-view descriptors for 3d point clouds
\[[paper](https://ieeexplore.ieee.org/document/9156894)\]
\[[code]()\]

* Learning multiview 3d point cloud registration
\[[paper](https://ieeexplore.ieee.org/document/9157740)\]
\[[code](https://github.com/zgojcic/3D_multiview_reg)\]

* Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting.
\[[paper](https://ieeexplore.ieee.org/document/10203551)\]
\[[code](https://github.com/WHU-USI3DV/SGHR)\]

<p id="s-2"></p>

### 2. Feature Enhancement

* FIRE-Net: Feature Interactive Representation for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9710761)\]
\[[code]()\]

* PCAM: Product of Cross-Attention Matrices for Rigid Registration of Point Clouds.
\[[paper](https://ieeexplore.ieee.org/document/9711218)\]
\[[code](https://github.com/valeoai/PCAM)\]

* RGM: Robust Point Cloud Registration Framework Based on Deep Graph Matching.
\[[paper](https://ieeexplore.ieee.org/document/9578566)\]
\[[code](https://github.com/fukexue/RGM)\]

* REGTR: End-to-end Point Cloud Correspondences with Transformers.
\[[paper](https://ieeexplore.ieee.org/document/9880077)\]
\[[code](https://github.com/yewzijian/RegTR)\]


\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]
\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]
\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]
\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]
\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]

\[[paper]()\]
\[[code]()\]
  




