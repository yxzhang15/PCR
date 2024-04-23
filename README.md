# A Collection of DL-based Point Cloud Registration Methods

This repository provides a summary of deep learning based point cloud registration algorithms. 

We classify registration algorithms into supervised and unsupervised, as follows.

[Supervised Point Cloud Registration Methods](#supervised)
 * [1. Descriptor Extraction](#s-1)
   * [1. 1 Two-view](#s-1-1)
   * [1. 2 Multi-view](#s-1-3)
 * [2. Correspondence Search](#s-3)
   * [2. 1 Partial-object](#s-3-1)
   * [2. 2 Full-object](#s-3-2)
 * [3. Outlier Filtering](#s-4)
 * [4. Transformation Parameter Estimation](#s-5)
 * [5. Optimization](#s-6)
   * [5. 1 ICP-based](#s-6-1)
   * [5. 2 Probabilistic-based](#s-6-2) 
 * [6. Multimodal](#s-7)

[Unsupervised Point Cloud Registration Methods](#unsupervised)
 * [1. Correspondence-free](#u-1)
 * [2. Correspondence-based](#u-2)
   
[Datasets](#dataset)

<p id="supervised"></p>

## Supervised Point Cloud Registration Methods

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

#### 1. 2 Multiview

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


<p id="s-3"></p>

### 3. Correspondence Search

<p id="s-3-1"></p>

#### 3. 1 Partial-object

* PointNetLK: Robust & Efficient Point Cloud Registration Using PointNet.
\[[paper](https://ieeexplore.ieee.org/document/8954359)\]
\[[code](https://github.com/hmgoforth/PointNetLK)\]

* Deep Closest Point: Learning Representations for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9009466)\]
\[[code](https://github.com/WangYueFt/dcp)\]

* PointNetLK Revisited.
\[[paper](https://ieeexplore.ieee.org/document/9577995)\]
\[[code](https://github.com/Lilac-Lee/PointNetLK_Revisited)\]

<p id="s-3-2"></p>

#### 3. 2 Partial-object

 * PRNet: Self-supervised Learning for Partial-to-partial Registration.
\[[paper](https://dl.acm.org/doi/10.5555/3454287.3455078)\]
\[[code](https://github.com/WangYueFt/prnet)\]

* RPM-Net: Robust Point Matching Using Learned Features.
\[[paper](https://ieeexplore.ieee.org/document/9157132)\]
\[[code](https://github.com/yewzijian/RPMNet)\]

* PREDATOR: Registration of 3D Point Clouds with Low Overlap.
\[[paper](https://ieeexplore.ieee.org/document/9577334)\]
\[[code](https://github.com/prs-eth/OverlapPredator)\]

* OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9709963)\]
\[[code](https://github.com/megvii-research/OMNet)\]

* STORM: Structure-Based Overlap Matching for Partial Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9705149)\]
\[[code]()\]

* FIRE-Net: Feature Interactive Representation for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9710761)\]
\[[code]()\]

* PCAM: Product of Cross-Attention Matrices for Rigid Registration of Point Clouds.
\[[paper](https://ieeexplore.ieee.org/document/9711218)\]
\[[code](https://github.com/valeoai/PCAM)\]

* REGTR: End-to-end Point Cloud Correspondences with Transformers.
\[[paper](https://ieeexplore.ieee.org/document/9880077)\]
\[[code](https://github.com/yewzijian/RegTR)\]


<p id="s-4"></p>

### 3. Outlier Filtering

* 3DRegNet: A Deep Neural Network for 3D Point Registration.
\[[paper](https://ieeexplore.ieee.org/document/9156303)\]
\[[code](https://github.com/3DVisionISR/3DRegNet)\]

* PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency.
\[[paper](https://ieeexplore.ieee.org/document/9578333)\]
\[[code](https://github.com/XuyangBai/PointDSC)\]

* DLF: Partial Point Cloud Registration With Deep Local Feature.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9866792)\]
\[[code]()\]

* 3D Registration with Maximal Cliques. 
\[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_3D_Registration_With_Maximal_Cliques_CVPR_2023_paper.pdf)\]
\[[code](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)\]

* Deep Hough Voting for Robust Global Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Deep_Hough_Voting_for_Robust_Global_Registration_ICCV_2021_paper.pdf)\]
\[[code](http://cvlab.postech.ac.kr/research/DHVR/)\]


<p id="s-5"></p>

### 4. Transformation Parameter Estimation

* DeTarNet: Decoupling Translation and Rotation by Siamese Network for Point Cloud Registration.
\[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19917)\]
\[[code](https://github.com/ZhiChen902/DetarNet)\]

* FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration
\[[paper](https://arxiv.org/abs/2106.03479)\]
\[[code](https://github.com/MegEngine/FINet)\]

<p id="s-6"></p>

### 5. Optimization

<p id="s-6-1"></p>

#### 5. 1 ICP-based

* Deep Closest Point: Learning Representations for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9009466)\]
\[[code](https://github.com/WangYueFt/dcp)\]

 * PRNet: Self-supervised Learning for Partial-to-partial Registration.
\[[paper](https://dl.acm.org/doi/10.5555/3454287.3455078)\]
\[[code](https://github.com/WangYueFt/prnet)\]

* IDAM: Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration.
\[[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690375.pdf)\]
\[[code](https://github.com/jiahaowork/idam)\]

<p id="s-6-2"></p>

#### 5. 2 Probabilistic-based

* DeepGMR: Learning Latent Gaussian Mixture Models for Registration.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_43)\]
\[[code](https://github.com/wentaoyuan/deepgmr)\]

* OGMM: Overlap-guided Gaussian Mixture Models for Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Mei_Overlap-Guided_Gaussian_Mixture_Models_for_Point_Cloud_Registration_WACV_2023_paper.pdf)\]
\[[code](https://github.com/gfmei/ogmm)\]

* Point Cloud Registration Based on Learning Gaussian Mixture Models With Global-Weighted Local Representations.
\[[paper](https://ieeexplore.ieee.org/document/10066279)\]
\[[code]()\]

* VBReg: Robust Outlier Rejection for 3D Registration with Variational Bayes.
\[[paper](https://ieeexplore.ieee.org/document/10204375)\]
\[[code](https://github.com/Jiang-HB/VBReg)\]

<p id="s-7"></p>

### 6. Multimodal
* PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_26)\]
\[[code]()\]

* PEAL: Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10203464)\]
\[[code](https://github.com/Gardlin/PEAL)\]

* ImLoveNet: Misaligned Image-supported Registration Network for Low-overlap Point Cloud Pairs.
\[[paper](https://dl.acm.org/doi/10.1145/3528233.3530744)\]
\[[code]()\]

* IMFNet: Interpretable Multimodal Fusion for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9919364)\]
\[[code](https://github.com/XiaoshuiHuang/IMFNet)\]

* GMF: General Multimodal Fusion Framework for Correspondence Outlier Rejection.
\[[paper](https://ieeexplore.ieee.org/document/9940574)\]
\[[code](https://github.com/XiaoshuiHuang/GMF)\]

<p id="unsupervised"></p>

## Unsupervised Point Cloud Registration Methods

<p id="u-1"></p>

### 1. Correspondence-free

* PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors.
\[[paper](https://dl.acm.org/doi/10.1007/978-3-030-01228-1_37)\]
\[[code](https://github.com/XuyangBai/PPF-FoldNet)\]

* PCRNet: Point Cloud Registration Network using PointNet Encoding
\[[paper](https://arxiv.org/abs/1908.07906)\]
\[[code](https://github.com/vinits5/pcrnet)\]

* UPCR: A Representation Separation Perspective to Correspondence-Free Unsupervised 3-D Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9638480)\]
\[[code]()\]

* UGMM: Unsupervised Point Cloud Registration by Learning Unified Gaussian Mixture Models.
\[[paper](https://ieeexplore.ieee.org/document/9790333)\]
\[[code](https://github.com/XiaoshuiHuang/UGMMREG)\]

* Research and Application on Cross-source Point Cloud Registration Method Based on Unsupervised Learning。
\[[paper](https://ieeexplore.ieee.org/abstract/document/10256428)\]
\[[code]()\]

<p id="u-2"></p>

### 2. Correspondence-based

* CEMNet: Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Sampling_Network_Guided_Cross-Entropy_Method_for_Unsupervised_Point_Cloud_Registration_ICCV_2021_paper.pdf)\]
\[[code](https://github.com/Jiang-HB/CEMNet)\]

* RIENet: Reliable Inlier Evaluation for Unsupervised Point Cloud Registration.
\[[paper](https://aaai.org/papers/02198-reliable-inlier-evaluation-for-unsupervised-point-cloud-registration/)\]
\[[code](https://github.com/supersyq/RIENet)\]

* UDPReg: Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10204460)\]
\[[code](https://github.com/gfmei/udpreg)\]

<p id="dataset"></p>

## Datasets

* Stanford: A Volumetric Method for Building Complex Models from Range Images.
\[[paper](https://dl.acm.org/doi/10.1145/237170.237269)\]
\[[code](http://www-graphics.stanford.edu/)\]

* ETH: Challenging Data Sets for Point Cloud Registration Algorithms.
\[[paper](https://dl.acm.org/doi/10.1177/0278364912458814)\]
\[[code](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration#:~:text=Challenging%20data%20sets%20for%20point%20cloud%20registration%20algorithms,%29%3A%20...%204%20Permissions%20...%205%20Contact%20)\]

* KITTI: Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite.
\[[paper](https://ieeexplore.ieee.org/document/6248074)\]
\[[code](www.cvlibs.net/datasets/kitti)\]

* ModelNet40: 3D ShapeNets: A Deep Representation for Volumetric Shapes.
\[[paper](https://ieeexplore.ieee.org/document/7298801)\]
\[[code](https://3dshapenets.cs.princeton.edu/)\]

* ShapeNet: An Information-Rich 3D Model Repository.
\[[paper](https://arxiv.org/abs/1512.03012)\]
\[[code](https://www.shapenet.org/)\]

* ICL-NUIM: A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM.
\[[paper](https://ieeexplore.ieee.org/document/6907054)\]
\[[code](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)\]

* 3DMatch: Learning the Matching of Local 3D Geometry in Range Scans.
\[[paper](https://techmatt.github.io/pdfs/3DMatch.pdf)\]
\[[code](https://3dmatch.cs.princeton.edu/)\]

* Apollo-SouthBay: L3-Net: Towards Learning Based LiDAR Localization for Autonomous Driving.
\[[paper](https://ieeexplore.ieee.org/document/8954371)\]
\[[code](https://developer.apollo.auto/southbay.html)\]

* ScanObjectNN. Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data.
\[[paper](https://ieeexplore.ieee.org/document/9009007)\]
\[[code](https://hkust-vgd.github.io/scanobjectnn/)\]

* WHU-TLS: Registration of Large-scale Terrestrial Laser Scanner Point Clouds: A Review and Benchmark. 
\[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300836)\]
\[[code](https://github.com/WHU-USI3DV/WHU-TLS)\]

* FlyingShapes: SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_SIRA-PCR_Sim-to-Real_Adaptation_for_3D_Point_Cloud_Registration_ICCV_2023_paper.pdf)\]
\[[code](https://github.com/Chen-Suyi/SIRA_Pytorch)\]



