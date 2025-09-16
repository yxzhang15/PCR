# A Collection of DL-based Point Cloud Registration Methods

This repository provides a summary of deep learning based point cloud registration algorithms. 

If you find this repository helpful, we would greatly appreciate it if you could cite our paper: https://doi.org/10.24963/ijcai.2024/922 and 
http://arxiv.org/abs/2404.13830.
```bibtex
@inproceedings{ijcai2024p922,
  title     = {A Comprehensive Survey and Taxonomy on Point Cloud Registration Based on Deep Learning},
  author    = {Zhang, Yu-Xin and Gui, Jie and Cong, Xiaofeng and Gong, Xin and Tao, Wenbing},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {8344--8353},
  year      = {2024},
  month     = {8}
}
```
```
@misc{zhang2025deeplearningbasedpointcloud,
      title={Deep Learning-Based Point Cloud Registration: A Comprehensive Survey and Taxonomy}, 
      author={Yu-Xin Zhang and Jie Gui and Baosheng Yu and Xiaofeng Cong and Xin Gong and Wenbing Tao and Dacheng Tao},
      year={2025},
      eprint={2404.13830},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.13830}, 
}
```

The taxonomy of deep learning-based point cloud registration algorithms is as follows.

[Pairwise Point Cloud Registration](#Pariwise)
 * [1. Supervised Methods](#supervised)
   * [1. 1 Registration Procedure](#s-1)
     * [1. 1. 1 Descriptor Extraction](#s-1-1)
     * [1. 1. 2 Overlap Prediction](#s-1-2)
     * [1. 1. 3 Similarity Matrix Optimization ](#s-1-3)
     * [1. 1. 4 Outlier Filtering](#s-1-4)
     * [1. 1. 5 Transformation Parameter Estimation](#s-1-5)
     * [1. 1. 6 Others ](#s-1-6)
   * [1. 2. Optimization Strategy](#s-2)
     * [1. 2. 1 GMM-Based](#s-2-1) 
     * [1. 2. 2 Bayesian-Based](#s-2-2) 
     * [1. 2. 3 Diffusion-Based](#s-2-3) 
     * [1. 2. 4 Multimodality-Based](#s-2-5)
     * [1. 2. 5 Pretrain-Based](#s-4-3)
   * [1. 3. Learning Paradigm](#s-3)
     * [1. 3. 1 Contrastive Learning](#s-3-1)
     * [1. 3. 2 Meta Learning](#s-3-2)
     * [1. 3. 3 Reinforcement Learning](#s-3-3)
   * [1. 4 Integration of Traditional Algorithms ](#s-4)
     * [1. 4. 1 Iterative Closest Point](#s-4-1)
     * [1. 4. 2 Robust Point Matching](#s-4-2)
     * [1. 4. 3 Lucas-Kanade](#s-4-3)

* [2. Unsupervised Point Cloud Registration Methods](#unsupervised)
   * [2. 1. Correspondence-free](#u-1)
     * [2. 1. 1 One-stage Registration](#u-1-1)
     * [2. 1. 2 Iterative Registration](#u-1-2)
   * [2. 2. Correspondence-based](#u-2)
     * [2. 2. 1 RGB-D](#u-2-1)
     * [2. 2. 2 Probability Model](#u-2-2)
     * [2. 2. 3 Descriptor-Based](#u-2-3)
     * [2. 2. 4 Geometric Consistency-Based](#u-2-4)

[Multi-Scan Point Cloud Registration](#multi)
 * [1. Multiview Registration Methods](#mv)
   * [1. 1. Meta-Shape-Based](#mv-1)
   * [1. 2. Pose Graph-Based](#mv-2)
   * [1. 3. Scene-Based](#mv-3)
 * [2. Multi-instance Registration Methods](#mi)
   * [2. 1. Correspondence Clustering](#mi-1)
   * [2. 2. Pose Clustering](#mi-2)
   * [2. 3. Instance-aware Modeling](#mi-3)

[Datasets](#dataset)

<p id="Pariwise"></p>

## Pairwise Point Cloud Registration

<p id="supervised"></p>

### 1 Supervised Methods

<p id="s-1"></p>

#### 1.1 Registration Procedure

<p id="s-1-1"></p>

##### 1.1.1 Descriptor Extraction
###### 1.1.1.1 Point-based
* Fully Convolutional Geometric Features.
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choy_Fully_Convolutional_Geometric_Features_ICCV_2019_paper.pdf)\] 
\[[code]()\] (2019, ICCV)

* D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bai_D3Feat_Joint_Learning_of_Dense_Detection_and_Description_of_3D_CVPR_2020_paper.pdf)\] 
\[[code](https://github.com/XuyangBai/D3Feat)\] (2020, CVPR)

* GeoTransformer: Fast and Robust Point Cloud Registration With Geometric Transformer.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10076895)\]
\[[code](https://github.com/qinzheng93/GeoTransformer)\] (2023, PAMI)

* RoITr: Rotation-Invariant Transformer for Point Cloud Matching.
\[[paper](https://ieeexplore.ieee.org/document/10204543)\]
\[[code](https://github.com/haoyu94/RoITr)\] (2023, CVPR)

* EGST: Enhanced Geometric Structure Transformer for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10319695)\] (2024, TVCG)

* Neighborhood Multi-Compound Transformer for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10485488)\] (2024, TCSVT)

* Dynamic Cues-assisted Transformer for Robust Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Dynamic_Cues-Assisted_Transformer_for_Robust_Point_Cloud_Registration_CVPR_2024_paper.pdf)\]
\[[code]()\] (2024, CVPR)

* PARE-Net: Position-Aware Rotation-Equivariant Networks for Robust Point Cloud Registration.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-031-72904-1_17)\] 
\[[code](https://github.com/yaorz97/PARENet)\] (2025, ECCV)

###### 1.1.1.2 Patch-based
* 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)\] 
\[[code](http://3dmatch.cs.princeton.edu)\] (2017, CVPR)

* PPFNet: Global Context Aware Local Features for Robust 3D Point Matching.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_PPFNet_Global_Context_CVPR_2018_paper.pdf)\] 
\[[code]()\] (2018, CVPR)

* Distinctive 3D local deep descriptors.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9411978)\]
\[[code](https://github.com/fabiopoiesi/dip)\] (2021, ICPR)

* GeDi: Learning General and Distinctive 3D Local Deep Descriptors for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9775606)\]
\[[code]()\] (2022, PAMI)

* You Only Hypothesize Once: Point Cloud Registration with Rotation-equivariant Descriptors.
\[[paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548023)\]
\[[code](https://github.com/HpWang-whu/YOHO)\] (2022, ACM MM)

* RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations.
\[[paper](https://ieeexplore.ieee.org/document/10044259)\]
\[[code](https://github.com/HpWang-whu/RoReg)\] (2023, PAMI)

* HA-TiNet: Learning a Distinctive and General 3D Local Descriptor for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10666793)\] 
\[[code](https://github.com/ahulq/HA-TiNet)\] (2024, TVCG)

###### 1.1.1.3 Voxel-based
* The Perfect Match: 3D Point Cloud Matching With Smoothed Densities.
\[[paper](https://ieeexplore.ieee.org/document/8954296)\]
\[[code](https://github.com/zgojcic/3DSmoothNet)\] (2019, CVPR)

* SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9577271)\]
\[[code](https://github.com/QingyongHu/SpinNet)\] (2021, CVPR)

* SphereNet: Learning a Noise-Robust and General Descriptor for Point Cloud Registration
\[[paper](https://ieeexplore.ieee.org/abstract/document/10356130)\]
\[[code](https://github.com/GuiyuZhao/SphereNet)\] (2023, TGRS)

<p id="s-1-2"></p>

##### 1.1.2 Overlap Prediction
* PREDATOR: Registration of 3D Point Clouds with Low Overlap.
\[[paper](https://ieeexplore.ieee.org/document/9577334)\]
\[[code](https://github.com/prs-eth/OverlapPredator)\] (2021, CVPR)

* OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9709963)\]
\[[code](https://github.com/megvii-research/OMNet)\] (2021, ICCV)

* RORNet: Partial-to-Partial Registration Network With Reliable Overlapping Representations
\[[paper](https://ieeexplore.ieee.org/abstract/document/10168979)\]
\[[code]()\] (2024, TNNLS)

* STORM: Structure-Based Overlap Matching for Partial Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9705149)\]
\[[code]()\] (2023, TPAMI)

* A Unified BEV Model for Joint Learning of 3D Local Features and Overlap Estimation
\[[paper](https://ieeexplore.ieee.org/abstract/document/10160492)\]
\[[code]()\] (2023, ICRA)

* Low Overlapping Point Cloud Registration Using Mutual Prior Based Completion Network
\[[paper](https://ieeexplore.ieee.org/abstract/document/10643008)\]
\[[code]()\] (2024, TIP)

<p id="s-1-3"></p>

##### 1.1.3 Similarity Matrix Optimization
 * PRNet: Self-supervised Learning for Partial-to-partial Registration.
\[[paper](https://dl.acm.org/doi/10.5555/3454287.3455078)\]
\[[code](https://github.com/WangYueFt/prnet)\] (2019, NIPS)

* FIRE-Net: Feature Interactive Representation for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9710761)\]
\[[code]()\] (2021, ICCV)

* One-Inlier is First: Towards Efficient Position Encoding for Point Cloud Registration.
\[[paper](https://papers.nips.cc/paper_files/paper/2022/hash/2e163450c1ae3167832971e6da29f38d-Abstract-Conference.html)\]
\[[code]()\] (2022, NIPS)

* End-to-end Learning the Partial Permutation Matrix for Robust 3D Point Cloud Registration.
\[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20250)\]
\[[code]()\] (2022, AAAI)

<p id="s-1-4"></p>

##### 1.1.4 Outlier Filtering
* Deep Hough Voting for Robust Global Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Deep_Hough_Voting_for_Robust_Global_Registration_ICCV_2021_paper.pdf)\]
\[[code](http://cvlab.postech.ac.kr/research/DHVR/)\] (2021, ICCV)

* DLF: Partial Point Cloud Registration With Deep Local Feature.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9866792)\]
\[[code](https://github.com/zhlSunLab/Partial-Point-Cloud-Registration-with-Deep-Local-Feature)\] (2023, TAI)

* PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency.
\[[paper](https://ieeexplore.ieee.org/document/9578333)\]
\[[code](https://github.com/XuyangBai/PointDSC)\] (2021, CVPR)

* SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust  Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_SC2-PCR_A_Second_Order_Spatial_Compatibility_for_Efficient_and_Robust_CVPR_2022_paper.pdf)\]
\[[code](https://github.com/ZhiChen902/SC2-PCR)\] (2022, CVPR)

* SC-PCR++: Rethinking the Generation and Selection for Efficient and Robust Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10115040)\]
\[[code]()\] (2023, TPAMI)

* 3D Registration with Maximal Cliques. 
\[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_3D_Registration_With_Maximal_Cliques_CVPR_2023_paper.pdf)\]
\[[code](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)\] (2023, CVPR)

* MAC: Maximal Cliques for 3D Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10636064)\]
\[[code](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)\] (2024, TPAMI)

* Hunter: Exploring High-Order Consistency for Point Cloud Registration With Severe Outliers.
\[[paper](https://ieeexplore.ieee.org/document/10246849)\]
\[[code]()\] (2023, TPAMI)

* Robust Point Cloud Registration via Random Network Co-Ensemble.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10430173)\]
\[[code](https://github.com/phdymz/RandPCR)\] (2024, TCSVT)

* Scalable 3D Registration via Truncated Entry-wise Absolute Residuals.
\[[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Scalable_3D_Registration_via_Truncated_Entry-wise_Absolute_Residuals_CVPR_2024_paper.html)\]
\[[code]()\] (2024, CVPR)

* FastMAC: Stochastic Spectral Sampling of Correspondence Graph.
\[[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_FastMAC_Stochastic_Spectral_Sampling_of_Correspondence_Graph_CVPR_2024_paper.html)\]
\[[code](https://github.com/Forrest-110/FastMAC)\] (2024, CVPR)

<p id="s-1-5"></p>

##### 1.1.5 Transformation Parameter Estimation
* Self-supervised Rigid Transformation Equivariance for Accurate 3D Point Cloud Registration
\[[paper](https://www.sciencedirect.com/science/article/pii/S0031320322002655)\]
\[[code]()\] (2022, PR)

* DeTarNet: Decoupling Translation and Rotation by Siamese Network for Point Cloud Registration.
\[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19917)\]
\[[code](https://github.com/ZhiChen902/DetarNet)\] (2022, AAAI)

* FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration
\[[paper](https://arxiv.org/abs/2106.03479)\]
\[[code](https://github.com/MegEngine/FINet)\] (2022, AAAI)

* Learning Compact Transformation Based on Dual Quaternion for Point Cloud Registration
\[[paper](https://ieeexplore.ieee.org/abstract/document/10381830)\]
\[[code]()\] (2024, TIM)

* Q-reg: End-to-end trainable point cloud registration with surface curvature
\[[paper](https://ieeexplore.ieee.org/abstract/document/10550460)\]
\[[code](https://github.com/jinsz/Q-REG)\] (2024, 3DV)

<p id="s-1-6"></p>

##### 1.1.6 Others
* DeepVCP: An End-to-End Deep Neural Network for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9009450)\]
\[[code](https://github.com/jundaozhilian/DeepVCP-PyTorch)\] (2019, ICCV)

* 3DRegNet: A Deep Neural Network for 3D Point Registration.
\[[paper](https://ieeexplore.ieee.org/document/9156303)\]
\[[code](https://github.com/3DVisionISR/3DRegNet)\] (2020, CVPR)

* HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9710151)\]
\[[code](https://ispc-group.github.io/hregnet)\] (2021, ICCV)

* Robust Point Cloud Registration Framework Based on Deep Graph Matching.
\[[paper]([https://ieeexplore.ieee.org/document/9710151](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Robust_Point_Cloud_Registration_Framework_Based_on_Deep_Graph_Matching_CVPR_2021_paper.pdf))\]
\[[code](https://github.com/fukexue/RGM)\] (2021, CVPR)

* REGTR: End-to-end Point Cloud Correspondences with Transformers.
\[[paper](https://ieeexplore.ieee.org/document/9880077)\]
\[[code](https://github.com/yewzijian/RegTR)\] (2022, CVPR)

* BUFFER: Balancing Accuracy, Efficiency, and Generalizability in Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ao_BUFFER_Balancing_Accuracy_Efficiency_and_Generalizability_in_Point_Cloud_Registration_CVPR_2023_paper.pdf)\]
\[[code](https://github.com/aosheng1996/BUFFER)\] (2023, CVPR)

* SACF-Net: Skip-Attention Based Correspondence Filtering Network for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10018266)\] (2023, TCSVT)

* PEAL: Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10203464)\]
\[[code](https://github.com/Gardlin/PEAL)\] (2023, CVPR)

* Full Transformer Framework for Robust Point Cloud Registration With Deep Information Interaction.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10122705)\]
\[[code](https://github.com/CGuangyan-BIT/DIT)\] (2023, TNNLS)

* RegFormer: An Efficient Projection-Aware Transformer Network for Large-Scale Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_RegFormer_An_Efficient_Projection-Aware_Transformer_Network_for_Large-Scale_Point_Cloud_ICCV_2023_paper.html)\]
\[[code](https://github.com/IRMVLab/RegFormer)\] (2023, ICCV)

* SPEAL: Skeletal Prior Embedded Attention Learning for Cross-Source Point Cloud Registration.
\[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28446)\] (2024, AAAI)

* A Consistency-Aware Spot-Guided Transformer for Versatile and Hierarchical Point Cloud Registration.
\[[paper](https://arxiv.org/pdf/2410.10295)\]
\[[code](https://github.com/RenlangHuang/CAST)\] (2024, NIPS)

<p id="s-2"></p>

#### 1.2 Optimization Strategy

<p id="s-2-1"></p>

##### 1.2.1 GMM-Based
* DeepGMR: Learning Latent Gaussian Mixture Models for Registration.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_43)\]
\[[code](https://github.com/wentaoyuan/deepgmr)\] (2020, ECCV)

* OGMM: Overlap-guided Gaussian Mixture Models for Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Mei_Overlap-Guided_Gaussian_Mixture_Models_for_Point_Cloud_Registration_WACV_2023_paper.pdf)\]
\[[code](https://github.com/gfmei/ogmm)\] (2023, WACV)

* Point Cloud Registration Based on Learning Gaussian Mixture Models With Global-Weighted Local Representations.
\[[paper](https://ieeexplore.ieee.org/document/10066279)\]
\[[code]()\] (2023, GRSL)

<p id="s-2-2"></p>

##### 1.2.2 Bayesian-Based
* VBReg: Robust Outlier Rejection for 3D Registration with Variational Bayes.
\[[paper](https://ieeexplore.ieee.org/document/10204375)\]
\[[code](https://github.com/Jiang-HB/VBReg)\] (2023, CVPR)

<p id="s-2-3"></p>

##### 1.2.3 Diffusion-Based
* Point Cloud Registration With Zero Overlap Rate and Negative Overlap Rate.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10238787)\]
\[[code]()\] (2023, RAL)

* Diffusionpcr: Diffusion models for robust multi-step point cloud registration.
\[[paper](https://arxiv.org/pdf/2312.03053)\]
\[[code]()\] (2023)

* PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations.
\[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/27775)\]
\[[code](https://github.com/AI-IT-AVs/PosDiffNet)\] (2024, CVPR)

* PointDifformer: Robust Point Cloud Registration with Neural Diffusion and Transformer.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10384401)\]
\[[code]()\] (2024, TGRS)

* Se(3) diffusion model-based point cloud registration for robust 6d object pose estimation.
\[[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/43069caa6776eac8bca4bfd74d4a476d-Paper-Conference.pdf)\]
\[[code](https://github.com/Jiang-HB/DiffusionReg)\] (2024, NIPS)

* Diff-Reg: Diffusion-Based Correspondence Searching in Doubly Stochastic Matrix Space for Point Cloud Registration.
\[[paper]([https://arxiv.org/pdf/2401.00436](https://link.springer.com/chapter/10.1007/978-3-031-73650-6_10))\]
\[[code](https://github.com/wuqianliang/Diff-Reg)\] (2025, ECCV)

<p id="s-2-4"></p>

##### 1.2.4 Multimodality-Based
* PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_26)\]
\[[code]()\] (2022, ECCV)

* ImLoveNet: Misaligned Image-supported Registration Network for Low-overlap Point Cloud Pairs.
\[[paper](https://dl.acm.org/doi/abs/10.1145/3528233.3530744)\]
\[[code]()\] (2022, SIGGRAPH)

* IMFNet: Interpretable Multimodal Fusion for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9919364)\]
\[[code](https://github.com/XiaoshuiHuang/IMFNet)\] (2022, RAL)

* GMF: General Multimodal Fusion Framework for Correspondence Outlier Rejection.
\[[paper](https://ieeexplore.ieee.org/document/9940574)\]
\[[code](https://github.com/XiaoshuiHuang/GMF)\] (2022, RAL)

* IGReg: Image-geometry-assisted Point Cloud Registration via Selective Correlation Fusion.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10443547)\]
\[[code]()\] (2024, TMM)

* SemReg: Semantics Constrained Point Cloud Registration.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-031-72940-9_17)\]
\[[code](https://github.com/SheldonFung98/SemReg.git)\] (2025, ECCV)

<p id="s-2-5"></p>

##### 1.2.5 Pretrain-Based.
* SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_SIRA-PCR_Sim-to-Real_Adaptation_for_3D_Point_Cloud_Registration_ICCV_2023_paper.pdf)\]
\[[code](https://github.com/Chen-Suyi/SIRA_Pytorch)\] (2023, ICCV)

* Zero-Shot Point Cloud Registration.
\[[paper](https://arxiv.org/pdf/2312.03032)\] (2023)

* PointRegGPT: Boosting 3D Point Cloud Registration using Generative Point-Cloud Pairs for Training.
\[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06787.pdf)\]
\[[code](https://github.com/Chen-Suyi/PointRegGPT)\] (2025, ECCV)

* Boosting 3D Point Cloud Registration by Transferring Multi-modality Knowledge.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10161411)\]
\[[code](https://github.com/phdymz/DBENet.git)\] (2023, ICRA)

<p id="s-3"></p>

#### 1.3 Learning Paradigm 

<p id="s-3-1"></p>

##### 1.3.1 Contrastive Learning
* SCRnet: A Spatial Consistency Guided Network using Contrastive Learning for Point Cloud Registration.
\[[paper](https://www.mdpi.com/2073-8994/14/1/140)\]
\[[code]()\] (2022)

* Density-invariant Features for Distant Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Density-invariant_Features_for_Distant_Point_Cloud_Registration_ICCV_2023_paper.pdf)\]
\[[code](https://github.com/liuQuan98/GCL)\] (2023, ICCV)

* UMERegRobust: Universal Manifold Embedding Compatible Features for Robust Point Cloud Registration.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-031-73016-0_21)\]
\[[code](https://github.com/yuvalH9/UMERegRobust)\] (2025, ECCV)

<p id="s-3-2"></p>

##### 1.3.2 Meta Learning
* Point-TTA: Test-Time Adaptation for Point Cloud Registration Using Multitask Meta-Auxiliary Learning.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hatem_Point-TTA_Test-Time_Adaptation_for_Point_Cloud_Registration_Using_Multitask_Meta-Auxiliary_ICCV_2023_paper.pdf)\] (2023, ICCV)

* 3D Meta-Registration: Learning to Learn Registration of 3D Point Clouds.
\[[paper](https://arxiv.org/abs/2010.11504)\] (2020)

<p id="s-3-3"></p>

##### 1.3.3 Reinforcement Learning
* Reagent: Point cloud registration using imitation and reinforcement learning.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bauer_ReAgent_Point_Cloud_Registration_Using_Imitation_and_Reinforcement_Learning_CVPR_2021_paper.pdf)\] (2021, CVPR)

* Point Cloud Registration via Heuristic Reward Reinforcement Learning.
\[[paper](https://www.mdpi.com/2571-905X/6/1/16)\]  (2023)


<p id="s-4"></p>

#### 1.4 Integration of Traditional Algorithms

<p id="s-4-1"></p>

##### 1.4.1 Iterative Closest Point
* Dcpcr: Deep compressed point cloud registration in large-scale outdoor environments.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9765365)\] (2022, RAL)

* Deep Closest Point: Learning Representations for Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9009466)\]
\[[code](https://github.com/WangYueFt/dcp)\] (2019, ICCV)

* Global-PBNet: A Novel Point Cloud Registration for Autonomous Driving.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9733274)\] (2022, TITS)

* IDAM: Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration.
\[[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690375.pdf)\]
\[[code](https://github.com/jiahaowork/idam)\] (2022, ECCV)

<p id="s-4-2"></p>

##### 1.4.2 Robust Point Matching
* RPM-Net: Robust Point Matching using Learned Features.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.pdf)\]
\[[code](https://github.com/yewzijian/RPMNet)\] (2020, CVPR)

<p id="s-4-3"></p>

##### 1.4.3 Lucas-Kanade
* PointNetLK: Robust & Efficient Point Cloud Registration Using PointNet.
\[[paper](https://ieeexplore.ieee.org/document/8954359)\]
\[[code](https://github.com/hmgoforth/PointNetLK)\] (2019, CVPR)

* PointNetLK Revisited.
\[[paper](https://ieeexplore.ieee.org/document/9577995)\]
\[[code](https://github.com/Lilac-Lee/PointNetLK_Revisited)\] (2021, CVPR)

<p id="unsupervised"></p>

### 2 Unsupervised Methods

<p id="u-1"></p>

#### 2.1 Correspondence-free 

<p id="u-1-1"></p>

##### 2.1.1 One-stage Registration
* 3D Point Cloud Registration with Multi-Scale Architecture and Unsupervised Transfer Learning.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9665863)\]
\[[code](https://github.com/humanpose1/MS-SVConv)\] (2021, 3DV)

* UPCR: A Representation Separation Perspective to Correspondence-Free Unsupervised 3-D Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/document/9638480)\] (2023, GRSL)

<p id="u-1-2"></p>

##### 2.1.2 Iterative Registration
* Deep-3DAligner: Unsupervised 3D Point Set Registration Network With Optimizable Latent Vector.
\[[paper](https://arxiv.org/pdf/2010.00321)\] (2020)

* Research and Application on Cross-source Point Cloud Registration Method Based on Unsupervised Learning。
\[[paper](https://ieeexplore.ieee.org/abstract/document/10256428)\] (2023, CYBER))

* Learning Discriminative Features via Multi-Hierarchical Mutual Information for Unsupervised Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10475373)\] (2024, TCSVT)

* PCRNet: Point Cloud Registration Network using PointNet Encoding.
\[[paper](https://arxiv.org/abs/1908.07906)\]
\[[code](https://github.com/vinits5/pcrnet)\] (2019)

<p id="u-2"></p>

#### 2.2 Correspondence-based

<p id="u-2-1"></p>

##### 2.2.1 RGB-D
* UnsupervisedR&R: Unsupervised Point Cloud Registration via Differentiable Rendering.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Banani_UnsupervisedRR_Unsupervised_Point_Cloud_Registration_via_Differentiable_Rendering_CVPR_2021_paper.pdf)\]
\[[code](https://github.com/mbanani/unsupervisedRR)\] (2021, CVPR)

* Improving RGB-D Point Cloud Registration by Learning Multi-scale Local Linear Transformation.
\[[paper](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_11)\]
\[[code](https://github.com/514DNA/LLT)\] (2022, ECCV)

* PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yuan_PointMBF_A_Multi-scale_Bidirectional_Fusion_Network_for_Unsupervised_RGB-D_Point_ICCV_2023_paper.pdf)\]
\[[code](https://github.com/phdymz/PointMBF)\] (2023, ICCV)

* NeRF-Guided Unsupervised Learning of RGB-D Registration.
\[[paper](https://arxiv.org/pdf/2405.00507)\]
\[[code]()\] (2024)

* Discriminative correspondence estimation for unsupervised rgb-d point cloud registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10716699)\]
\[[code]()\] (2024, TCSVT)

<p id="u-2-2"></p>

##### 2.2.2 Probability Model
* Planning with Learned Dynamic Model for Unsupervised Point Cloud Registration.
\[[paper](https://www.ijcai.org/proceedings/2021/0107.pdf)\] (2021, IJCAI)

* CEMNet: Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Sampling_Network_Guided_Cross-Entropy_Method_for_Unsupervised_Point_Cloud_Registration_ICCV_2021_paper.pdf)\]
\[[code](https://github.com/Jiang-HB/CEMNet)\] (2021, ICCV)

* UGMM: Unsupervised Point Cloud Registration by Learning Unified Gaussian Mixture Models.
\[[paper](https://ieeexplore.ieee.org/document/9790333)\]
\[[code](https://github.com/XiaoshuiHuang/UGMMREG)\] (2022, RAL)

* Overlap Bias Matching is Necessary for Point Cloud Registration.
\[[paper](https://arxiv.org/pdf/2308.09364)\] (2023)

* UDPReg: Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10204460)\]
\[[code](https://github.com/gfmei/udpreg)\] (2023, CVPR)

<p id="u-2-3"></p>

##### 2.2.3 Descriptor-based
* PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors.
\[[paper](https://dl.acm.org/doi/10.1007/978-3-030-01228-1_37)\]
\[[code](https://github.com/XuyangBai/PPF-FoldNet)\] (2018, ECCV)

* DeepUME: Learning the Universal Manifold Embedding for Robust Point Cloud Registration.
\[[paper](https://arxiv.org/pdf/2112.09938)\]
\[[code](https://github.com/langnatalie/DeepUME)\] (2021, BMVC)

* Corrnet3d: Unsupervised end-to-end learning of dense correspondence for 3d point clouds.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zeng_CorrNet3D_Unsupervised_End-to-End_Learning_of_Dense_Correspondence_for_3D_Point_CVPR_2021_paper.pdf)\]
\[[code](https://github.com/ZENGYIMING-EAMON/CorrNet3D)\] (2021, CVPR)

* R-PointHop: A Green, Accurate, and Unsupervised Point Cloud Registration Method.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9741387)\]
\[[code](https://github.com/pranavkdm/R-PointHop)\] (2022, TIP)

* GTINet: Global Topology-Aware Interactions for Unsupervised Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10440120)\]
\[[code]()\] (2024, TCSVT)

<p id="u-2-4"></p>

##### 2.2.4 Geometric Consistency-based
* RIENet: Reliable Inlier Evaluation for Unsupervised Point Cloud Registration.
\[[paper](https://aaai.org/papers/02198-reliable-inlier-evaluation-for-unsupervised-point-cloud-registration/)\]
\[[code](https://github.com/supersyq/RIENet)\] (2022, AAAI)

* RegiFormer: Unsupervised Point Cloud Registration via Geometric Local-to-Global Transformer and Self Augmentation.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10613860)\] (2024, TGRS)

* Extend Your Own Correspondences: Unsupervised Distant Point Cloud Registration by Progressive Distance Extension.
\[[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Extend_Your_Own_Correspondences_Unsupervised_Distant_Point_Cloud_Registration_by_CVPR_2024_paper.pdf)\]
\[[code](https://github.com/liuQuan98/EYOC)\] (2024, CVPR)

* Inlier Confidence Calibration for Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Inlier_Confidence_Calibration_for_Point_Cloud_Registration_CVPR_2024_paper.pdf)\] (2024, CVPR)

* Mining and Transferring Feature-Geometry Coherence for Unsupervised Point Cloud Registration.
\[[paper](https://arxiv.org/abs/2411.01870)\]
\[[code](https://github.com/kezheng1204/INTEGER)\] (2024, NIPS)

<p id="multi"></p>

## Multi-Scan Point Cloud Registration

<p id="multi"></p>

### 1. Multiview Registration Methods

<p id="mv-1"></p>

#### 1.1 Meta-Shape-Based

* Incremental Multiview Point Cloud Registration with Two-stage Candidate Retrieval.
\[[paper](https://www.sciencedirect.com/science/article/pii/S0031320325003656?casa_token=2ARbc6TogsoAAAAA:I7940dj7OE2tTWs-XnMoWD4NaiSX4BkSDyjRBGg48PeoSLS5fryLFXorohEkvg_pyMgdB6OiFnVS)\]
\[[code]()\] (2025, PR)

<p id="mv-2"></p>

#### 1.2 Pose Graph-Based

* Matching Distance and Geometric Distribution Aided Learning Multiview Point Cloud Registration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10669215?casa_token=xaY0jZkS09QAAAAA:8hBWGBruDJxZCInOYZncLizPUQwRC69ac2zYCvqStM-BemH6xlVZ2prcPpWHT5J4eKLYhQ20pmcY)\]
\[[code]()\] (2024, RAL)

* Registration of Multiview Point Clouds with Unknown Overlap.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10179137?casa_token=HGr2ZFmS26kAAAAA:_1S2Jwc-0zRqMhelPJ0XD4-2BgQcReePLKdnFCv-U9758_HpOmbj2ujRpNcR4-TPoOEL0-nwcDUm)\]
\[[code]()\] (2023, TMM)

* Learning Transformation Synchronization.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Learning_Transformation_Synchronization_CVPR_2019_paper.html)\]
\[[code]()\] (2019, CVPR)

* Learning Iterative Robust Transformation Synchronization.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9665877?casa_token=-Vh8vUdYG2oAAAAA:5ySL1XEe-4h4d9F8ud_60pbCt8DJ9gkZ_K1viGUiLS5p3sgqwPIvg5Ji0hxw0oUJKrRs61kLhrF7)\]
\[[code]()\] (2021, 3DV)

* Learning Multiview 3D Point Cloud Registration
\[[paper](https://ieeexplore.ieee.org/document/9157740)\]
\[[code](https://github.com/zgojcic/3D_multiview_reg)\] (2020, CVPR)

* Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting.
\[[paper](https://ieeexplore.ieee.org/document/10203551)\]
\[[code](https://github.com/WHU-USI3DV/SGHR)\] (20223, CVPR)

<p id="mv-3"></p>

#### 1.3 Scene-Based
* DeepMapping: Unsupervised Map Estimation from Multiple Point Clouds.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Ding_DeepMapping_Unsupervised_Map_Estimation_From_Multiple_Point_Clouds_CVPR_2019_paper.html)\]
\[[code](https://ai4ce.github.io/DeepMapping/)\] (2019, CVPR)

* Deepmapping2: Self-supervised Large-scale Lidar Map Optimization.
\[[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_DeepMapping2_Self-Supervised_Large-Scale_LiDAR_Map_Optimization_CVPR_2023_paper.html)\]
\[[code](https://ai4ce.github.io/DeepMapping2/)\] (2023, CVPR)

* Multiview Point Cloud Registration via Optimization in An Autoencoder Latent Space.
\[[paper](https://ieeexplore.ieee.org/abstract/document/10989630?casa_token=2oYqKuWEVtwAAAAA:2BWc3pRt1RcnRxaHlzPjWsZXr190vbPrjZc4H3J-43AUTCJuuPszej6zfYRKZi7m0U916M5IvEvo)\]
\[[code](github.com/pypolar/polar)\] (2025, TIP)

<p id="mi"></p>

### 2. Multiview Registration Methods

<p id="mi-1"></p>

#### 2.1 Correspondence Clustering
* Multi-instance Point Cloud Registration by Efficient Correspondence Clustering.
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Multi-Instance_Point_Cloud_Registration_by_Efficient_Correspondence_Clustering_CVPR_2022_paper.html)\]
\[[code](https://github.com/Gilgamesh666666/Multi-instance-Point-Cloud-Registration-by-Efficient-Correspondence-Clustering)\] (2022, CVPR)

* PointCLM: A Contrastive Learning-based Framework for Multi-instance Point Cloud Registration.
\[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690586.pdf)\]
\[[code](http://github.com/phdymz/PointCLM)\] (2022, ECCV)
<p id="mi-2"></p>

#### 2.2 Pose Clustering
* PointMC: Multi-instance Point Cloud Registration based on Maximal Cliques.
\[[paper](https://openreview.net/forum?id=0JV5WpLQgv)\]
\[[code]()\] (2024, ICML)

<p id="mi-3"></p>

#### 2.3 Instance-aware Modeling

* Learning Instance-Aware Correspondences for Robust Multi-Instance Point Cloud Registration in Cluttered Scenes 
\[[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Learning_Instance-Aware_Correspondences_for_Robust_Multi-Instance_Point_Cloud_Registration_in_CVPR_2024_paper.html)\]
\[[code](https://github.com/zhiyuanYU134/MIRETR)\] (2024, CVPR)

*  3D Focusing-and-matching Network for Multi-instance Point Cloud Registration.
\ [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cc52950239c3129464b0a6e379e2a9b0-Abstract-Conference.html)\]
\[[code](https://github.com/zlynpu/3DFMNet)\] (2024, NIPS)

<p id="dataset"></p>

## Datasets

* ETH: Challenging Data Sets for Point Cloud Registration Algorithms.
\[[paper](https://dl.acm.org/doi/10.1177/0278364912458814)\]
\[[code](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration#:~:text=Challenging%20data%20sets%20for%20point%20cloud%20registration%20algorithms,%29%3A%20...%204%20Permissions%20...%205%20Contact%20)\]

* KITTI: Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite.
\[[paper](https://ieeexplore.ieee.org/document/6248074)\]
\[[code](www.cvlibs.net/datasets/kitti)\]

* ICL-NUIM: A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM.
\[[paper](https://ieeexplore.ieee.org/document/6907054)\]
\[[code](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)\]

* ModelNet40: 3D ShapeNets: A Deep Representation for Volumetric Shapes.
\[[paper](https://ieeexplore.ieee.org/document/7298801)\]
\[[code](https://3dshapenets.cs.princeton.edu/)\]

* ShapeNet: An Information-Rich 3D Model Repository.
\[[paper](https://arxiv.org/abs/1512.03012)\]
\[[code](https://www.shapenet.org/)\]

* RedWood: A large dataset of object scans
\[[paper](https://arxiv.org/pdf/1602.02481)\]
\[[code](http://redwood-data.org/3dscan)\]

* 3DMatch: Learning the Matching of Local 3D Geometry in Range Scans.
\[[paper](https://techmatt.github.io/pdfs/3DMatch.pdf)\]
\[[code](https://3dmatch.cs.princeton.edu/)\]

* Oxford RobotCar: 1 year, 1000 km: The oxford robotcar dataset
\[[paper](https://journals.sagepub.com/doi/abs/10.1177/0278364916679498)\]
\[[code](http://robotcar-dataset.robots.ox.ac.uk)\]

* ScanObjectNN. Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data.
\[[paper](https://ieeexplore.ieee.org/document/9009007)\]
\[[code](https://hkust-vgd.github.io/scanobjectnn/)\]

* WHU-TLS: Registration of Large-scale Terrestrial Laser Scanner Point Clouds: A Review and Benchmark. 
\[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300836)\]
\[[code](https://github.com/WHU-USI3DV/WHU-TLS)\]

* Nuscenes: A multimodal dataset for autonomous driving
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)\]
\[[code](https://github.com/nutonomy/nuscenes-devkit)\]

* MVP-RG: Robust partial-to-partial point cloud registration in a full rang
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Variational_Relational_Point_Completion_Network_CVPR_2021_paper.pdf)\]
\[[code](https://github.com/paul007pl/MVP_Benchmark/tree/main/registration)\]

* FlyingShapes: SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration.
\[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_SIRA-PCR_Sim-to-Real_Adaptation_for_3D_Point_Cloud_Registration_ICCV_2023_paper.pdf)\]
\[[code](https://github.com/Chen-Suyi/SIRA_Pytorch)\]
