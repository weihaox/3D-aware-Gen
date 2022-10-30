<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
<meta name="generator" content="jemdoc, see http://jemdoc.jaboc.net/" />
<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
<link rel="stylesheet" href="jemdoc.css" type="text/css" />
</head>
<body>
<div id="layout-content">
<div id="toptitle">
  <h1 align="center">Awesome 3D-aware Image Synthesis &ndash; Papers, Codes and Datasets</h1>
</div>

Table of contents
-----------------

1.  **Introduction**
    
2.  **Survey paper**
    
3.  **Datasets**
    
    1.  multi-view image collections
        
    2.  Single-view image collections
        
4.  **3D Control of 2D Generative Models**
    
    1.  3D Control Latent Directions
        
    2.  3D Parameters as Controls
        
    3.  3D Prior Knowledge as Constraints
        
5.  **3D Novel View Synthesis from Multiple Views**
    
    1.  Neural Scene Representation
        
    2.  Generalization
        
    3.  Speed up
        
    4.  From Constrained Environmental Conditions to In-the-wild
        
        1.  Few images
            
        2.  Pose-free
            
        3.  Varying appearance
            
        4.  Large-scale scene
            
        5.  Dynamic scene
            
6.  **3D Generative Models from Single Views**
    
    1.  Unconditional 3D Generative Models
        
    2.  Conditional 3D Generative Models
        
7.  **3D-aware Video Synthesis**
    

Introduction
------------

This homepage lists some representative papers/codes/datasets all about **[3D-aware image synthesis](https://weihaox.github.io/awesome-3D-aware-synthesis/)**. We aim to constantly update the latest relevant papers and help the community track this topic. Please feel free to join us and [contribute](https://github.com/weihaox/awesome-3D-aware-synthesis/blob/main/CONTRIBUTE.md) to the project. If you have any questions, please feel free to contact [Weihao Xia](http://weihaox.github.io).

Survey paper
------------

*   [A Survey on 3D-aware Image Synthesis](https://arxiv.org/abs/2210.14267)  
    Weihao Xia and Jing-Hao Xue. _arXiv: 2210.14267_, 2022.
    
*   [Deep Generative Models on 3D Representations: A Survey](https://arxiv.org/abs/2210.15663)  
    Zifan Shi, Sida Peng, Yinghao Xu, Yiyi Liao, Yujun Shen. _arXiv: 2210.15663_, 2022.

Datasets
------------------

Summary of popular 3D-aware image synthesis datasets.

### Multi-view image collections

The images are rendered or collected according to different experimental settings, such as Synthetic-NeRF dataset, the DTU dataset, and the Tanks and Temples dataset for general purposes, the crowded Phototourism dataset for varying lighting conditions, the Blender Forward Facing (BLEFF) dataset to benchmark camera parameter estimation and novel view synthesis quality, and the San Francisco Alamo Square Dataset for large-scale scenes.

Examples of multi-view image datasets.

| **dataset**       | **published in**     |     **# scene**       |   **# samples per scene**   |   **range (m × m)}**.  |   **resolution**    |   **keyword**          |
|:-----------------:|:--------------------:|:---------------------:|:---------------------------:|:----------------------:|:-------------------:|:-----------------------:|
| DeepVoxels        | CVPR 2019            | 4 simple objects      | 479 / 1,000                 | \                      | 512 × 512           | synthetic, 360 degree   |
| NeRF Synthetics   | ECCV 2020            | 8 complex objects     | 100 / 200                   | \                      | 800 ×800            | synthetic, 360 degree   |
| NeRF Captured     | ECCV 2020            | 8 complex scenes      | 20-62                       | a few                  | 1,008 × 756         | real, forward-facing    |
| DTU               | CVPR 2014            | 124 scenes            | 49 or 64                    | a few to thousand      | 1,600 × 1,200       | often used in few-views |
| Tanks and Temples | CVPR 2015            | 14 objects and scenes | 4,395 - 21,871              | dozen to thousand      | 8-megapixel         | real, large-scale       |
| Phototourism      | IJCV 2021            | 6 landmarks           | 763-2,000                   | dozen to thousand      | 564-1,417 megapixel | varying illumination    |
| Alamo Square      | CVPR 2022            | San Francisco         | 2,818,745                   | 570 × 960              | 1,200 × 900         | real, large-scale       |


### Single-view image collections

Summary of popular single-view image datasets organized by their major categories and sorted by their popularity.

| **dataset** | **year**     | **category**           | **# samples** | **resolution**   | **keyword**            |
|:-----------:|:------------:|:----------------------:|:-------------:|:------------------:|:----------------------:|
| FFHQ        | CVPR 2019    | Human Face             | 70k           | 1024 × 1024    | single simple-shape   |
| AFHQ        | CVPR 2020    | Cat, Dog, and Wildlife | 15k           | 512  × 512     | single simple-shape    |
| CompCars    | CVPR 2015    | Real Car               | 136K          | 256  × 256     | single simple-shape    |
| CARLA       | CoRL 2017    | Synthetic Car          | 10k           | 128  × 128     | single simple-shape    |
| CLEVR       | CVPR 2017    | Objects                | 100k          | 256  × 256     | multiple, simple-shape |
| LSUN        | 2015         | Bedroom                | 300K          | 256  × 256     | single, simple-shape   |
| CelebA      | ICCV 2015    | Human Face             | 200k          | 178  × 218     | single simple-shape    |
| CelebA-HQ   | ICLR 2018    | Human Face             | 30k           | 1024 × 1024    | single, simple-shape   |
| MetFaces    | NeurIPS 2020 | Art Face               | 1336          | 1024 × 1024    | single, simple-shape   |
| M-Plants    | NeurIPS 2022 | Variable-Shape         | 141,824       | 256  × 256     | single, variable-shape |
| M-Food      | NeurIPS 2022 | Variable-Shape         | 25,472        | 256  × 256     | single, variable-shape |


## 3D Control of 2D Generative Models
### 3D Control Latent Directions

* **SeFa: Closed-Form Factorization of Latent Semantics in GANs.**<br>
*Yujun Shen, Bolei Zhou.*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2007.06600)] [[Github](https://github.com/genforce/sefa)] [[Project](https://genforce.github.io/sefa/)]

* **On the "steerability" of generative adversarial networks.**<br>
*Ali Jahanian, Lucy Chai, Phillip Isola.*<br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1907.07171)] [[Project](https://ali-design.github.io/gan_steerability/)]

* **GANSpace: Discovering Interpretable GAN Controls.**<br>
*Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, Sylvain Paris.*<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2004.02546)] [[Github](https://github.com/harskish/ganspace)]

* **Interpreting the Latent Space of GANs for Semantic Face Editing.**<br>
*[Yujun Shen](http://shenyujun.github.io/), [Jinjin Gu](http://www.jasongt.com/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml), [Bolei Zhou](http://bzhou.ie.cuhk.edu.hk/).*<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1907.10786)] [[Project](https://genforce.github.io/interfacegan/)] [[Github](https://github.com/genforce/interfacegan)]

* **nsupervised Discovery of Interpretable Directions in the GAN Latent Space.**<br>
*Andrey Voynov, Artem Babenko.*<br>
ICML 2020. [[PDF](https://arxiv.org/abs/2002.03754)] [[Github](https://github.com/anvoynov/GANLatentDiscovery)]

### 3D Parameters as Controls

* **StyleRig: Rigging StyleGAN for 3D Control over Portrait Images.**<br>
*Ayush Tewari, Mohamed Elgharib, Gaurav Bharaj, Florian Bernard, Hans-Peter Seidel, Patrick Pérez, Michael Zollhöfer, Christian Theobalt.*<br>
CVPR 2020 (oral). [[PDF](https://arxiv.org/abs/2004.00121)] [[Project](https://gvv.mpi-inf.mpg.de/projects/StyleRig/)]

* **DiscoFaceGAN: Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning.**<br>
*Yu Deng, Jiaolong Yang, Dong Chen, Fang Wen, Xin Tong.*<br>
CVPR 2020. [[PDF](https://arxiv.org/pdf/2004.11660.pdf)] [[Github](https://github.com/microsoft/DiscoFaceGAN)] 

* **PIE: Portrait Image Embedding for Semantic Control.**<br> 
*[A. Tewari](http://people.mpi-inf.mpg.de/~atewari/), M. Elgharib, M. BR, F. Bernard, H-P. Seidel, P. P‌érez, M. Zollhöfer, C.Theobalt.*<br> 
SIGGRAPH Asia 2020. [[PDF](http://gvv.mpi-inf.mpg.de/projects/PIE/data/paper.pdf)] [[Project](http://gvv.mpi-inf.mpg.de/projects/PIE/)]

* **CONFIG: Controllable Neural Face Image Generation.**<br>
*Marek Kowalski, Stephan J. Garbin, Virginia Estellers, Tadas Baltrušaitis, Matthew Johnson, Jamie Shotton.*<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2005.02671)] [[Github](https://github.com/microsoft/ConfigNet)]

* **GAN-Control: Explicitly Controllable GANs.**<br>
*Alon Shoshan, Nadav Bhonker, Igor Kviatkovsky, Gerard Medioni.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2101.02477)] [[Project](https://alonshoshan10.github.io/gan_control/)]

* **3D-FM GAN: Towards 3D-Controllable Face Manipulation.**<br>
*[Yuchen Liu](https://lychenyoko.github.io/), Zhixin Shu, Yijun Li, Zhe Lin, Richard Zhang, and Sun-Yuan Kung.*<br>
ECCV 2022. [[PDF](https://arxiv.org/abs/2208.11257)] [[Project](https://lychenyoko.github.io/3D-FM-GAN-Webpage/)]


### 3D Prior Knowledge as Constraints

* **Generative Image Modeling using Style and Structure Adversarial Networks.**<br>
*Xiaolong Wang, Abhinav Gupta.*<br>
ECCV 2016. [[PDF](https://arxiv.org/abs/1603.05631)]

* **3D Shape Induction from 2D Views of Multiple Objects.**<br>
*Matheus Gadelha, Subhransu Maji, Rui Wang.*<br>
3DV 2017. [[PDF](https://arxiv.org/abs/1612.05872)] [[Project](http://mgadelha.me/prgan/index.html)]

* **Visual Object Networks: Image Generation with Disentangled 3D Representation.**<br>
*Jun-Yan Zhu, Zhoutong Zhang, Chengkai Zhang, Jiajun Wu, Antonio Torralba, Joshua B. Tenenbaum, William T. Freeman.*<br>
NeurIPS 2018. [[PDF](https://arxiv.org/abs/1812.02725)] [[Project](http://von.csail.mit.edu/)] [[Github](https://github.com/junyanz/VON)]

* **RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis.**<br>
*Atsuhiro Noguchi, Tatsuya Harada.*<br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1909.12573)] [[Github](https://github.com/nogu-atsu/RGBD-GAN)]

* **NGP: Towards a Neural Graphics Pipeline for Controllable Image Generation.**<br>
*Xuelin Chen, Daniel Cohen-Or, Baoquan Chen, Niloy J. Mitra.*<br>
Eurographics 2021. [[PDF](https://arxiv.org/abs/2006.10569)] [[Github](http://geometry.cs.ucl.ac.uk/projects/2021/ngp)]

* **Lifting 2D StyleGAN for 3D-Aware Face Generation.**<br>
*[Yichun Shi](https://seasonsh.github.io/), Divyansh Aggarwal, [Anil K. Jain](http://www.cse.msu.edu/~jain/).*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.13126)]

* **3D-Aware Indoor Scene Synthesis with Depth Priors.**<br>
*Zifan Shi, Yujun Shen, Jiapeng Zhu, Dit-Yan Yeung, Qifeng Chen.*<br>
ECCV 2022 (oral). [[PDF](https://arxiv.org/abs/2202.08553)] [[Project](https://vivianszf.github.io/depthgan/)] [[Github](https://github.com/vivianszf/depthgan)] 

## 3D Novel View Synthesis from Multiple Views

### Neural Scene Representation

* **DeepVoxels: Learning Persistent 3D Feature Embeddings.**<br>
*Vincent Sitzmann, Justus Thies, Felix Heide, Matthias Nießner, Gordon Wetzstein, Michael Zollhöfer.*<br>
CVPR 2019 (Oral). [[Project](http://vsitzmann.github.io/deepvoxels/)] [[PDF](https://arxiv.org/abs/1812.01024)] [[Code](https://github.com/vsitzmann/deepvoxels)]

* **Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations.**<br>
*[Vincent Sitzmann](https://vsitzmann.github.io/), Michael Zollhöfer, Gordon Wetzstein.*<br>
NeurIPS 2019 (Oral, Honorable Mention "Outstanding New Directions").
[[PDF](http://arxiv.org/abs/1906.01618)] [[Project](https://github.com/vsitzmann/scene-representation-networks)] [[Github](https://github.com/vsitzmann/scene-representation-networks)] [[Dataset](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90?usp=sharing)]

* **Differentiable Volumetric Rendering (DVR): Learning Implicit 3D Representations without 3D Supervision.**<br>
*Michael Niemeyer, Lars Mescheder, Michael Oechsle, Andreas Geiger.*<br>
CVPR 2020. [[PDF](http://www.cvlibs.net/publications/Niemeyer2020CVPR.pdf)] [[Github](https://github.com/autonomousvision/differentiable_volumetric_rendering)]

* **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.**<br>
*[Ben Mildenhall](http://people.eecs.berkeley.edu/~bmild/), [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/), [Matthew Tancik](http://www.matthewtancik.com/), [Jonathan T. Barron](https://jonbarron.info/), [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/), [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html).*<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2003.08934)] [[Project](http://tancik.com/nerf)] [[Gtihub-Tensorflow](https://github.com/bmild/nerf)] [[krrish94-PyTorch](https://github.com/krrish94/nerf-pytorch)] [[yenchenlin-PyTorch](https://github.com/yenchenlin/nerf-pytorch)]

* **Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields.**<br>
*[Jonathan T. Barron](https://jonbarron.info/), [Ben Mildenhall](https://bmild.github.io/), [Matthew Tancik](https://www.matthewtancik.com/), [Peter Hedman](https://phogzone.com/cv.html), [Ricardo Martin-Brualla](http://ricardomartinbrualla.com/), [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/).*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.13415)] [[Project](http://jonbarron.info/mipnerf)]

* **Light Field Networks (LFNS): Neural Scene Representations with Single-Evaluation Rendering.**<br>
*Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, Fredo Durand.*<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2106.02634)] [[Project](https://vsitzmann.github.io/lfns/)]

* **Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations.**<br>
*Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas Funkhouser, Andrea Tagliasacchi.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.13152)]

### Speed up

* **NSVF: Neural Sparse Voxel Fields.**<br>
*[Lingjie Liu](https://lingjie0206.github.io/), Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, Christian Theobalt.*<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2007.11571)] [[Project](https://lingjie0206.github.io/papers/NSVF/)] [[Code](https://github.com/facebookresearch/NSVF)]

* **AutoInt: Automatic Integration for Fast Neural Volume Rendering.**<br>
*David B. Lindell, Julien N. P. Martel, Gordon Wetzstein.*<br>
CVPR 2021 (oral). [[PDF](https://arxiv.org/abs/2012.01714)] [[Project](http://www.computationalimaging.org/publications/automatic-integration/)]

* **KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs.**<br>
*Christian Reiser, Songyou Peng, Yiyi Liao, Andreas Geiger.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.13744)] [[Github](https://github.com/creiser/kilonerf)]

* **FastNeRF: High-Fidelity Neural Rendering at 200FPS.**<br>
*Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.10380)]

* **PlenOctrees for Real-time Rendering of Neural Radiance Fields.**<br>
*[Alex Yu](https://alexyu.net/), [Ruilong Li](https://www.liruilong.cn/), [Matthew Tancik](https://www.matthewtancik.com/), [Hao Li](https://www.hao-li.com/), [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/).*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.14024)] [[Project](https://alexyu.net/plenoctrees/)] [[Github](https://github.com/sxyu/plenoctree)]

* **Baking Neural Radiance Fields for Real-Time View Synthesis.**<br>
*[Peter Hedman](https://phogzone.com/), [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/), [Ben Mildenhall](https://bmild.github.io/), [Jonathan T. Barron](https://jonbarron.info/), [Paul Debevec](https://www.pauldebevec.com/).*<br>
ICCV 2021 (oral). [[PDF](https://arxiv.org/abs/2103.14645)] [[Project](https://nerf.live/)]

* **DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering.**<br>
*[Liwen Wu](https://lwwu2.github.io/), [Jae Yong Lee](https://jyl.kr/), [Anand Bhattad](https://anandbhattad.github.io/), [Yuxiong Wang](https://yxw.web.illinois.edu/), [David A. Forsyth](http://luthuli.cs.uiuc.edu/~daf/).*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.10427)] [[Project](https://lwwu2.github.io/diver/)] [[Github](https://github.com/lwwu2/diver)]

* **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.**<br>
*[Thomas Müller](https://tom94.net/), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller).*<br>
ACM Transactions on Graphics (SIGGRAPH) 2022. [[PDF](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)] [[Project](https://nvlabs.github.io/instant-ngp)][[Github](https://github.com/NVlabs/instant-ngp)]


### From Constrained Environmental Conditions to In-the-wild

**Few images**

* **GRF: Learning a General Radiance Field for 3D Representation and Rendering.**<br>
*Alex Trevithick, Bo Yang.*<br>
ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Trevithick_GRF_Learning_a_General_Radiance_Field_for_3D_Representation_and_ICCV_2021_paper.html)]

* **pixelNeRF: Neural Radiance Fields from One or Few Images.**<br>
*[Alex Yu](https://alexyu.net/), Vickie Ye, Matthew Tancik, Angjoo Kanazawa.*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.02190)] [[Project](https://alexyu.net/pixelnerf)]

* **IBRNet: Learning Multi-View Image-Based Rendering.**<br>
*Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, Thomas Funkhouser.*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2102.13090)] [[Project](https://ibrnet.github.io/)]

* **MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo.**<br>
*[Anpei Chen](https://apchenstu.github.io/), [Zexiang Xu](http://cseweb.ucsd.edu/~zex014/), Fuqiang Zhao, Xiaoshuai Zhang, [Fanbo Xiang](https://www.fbxiang.com/), [Jingyi Yu](http://vic.shanghaitech.edu.cn/vrvc/en/people/), [Hao Su](https://cseweb.ucsd.edu/~haosu/).*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.15595)] [[Project](https://apchenstu.github.io/mvsnerf/)] [[Github](https://github.com/apchenstu/mvsnerf)]

* **CodeNeRF: Disentangled Neural Radiance Fields for Object Categories.**<br>
*Wonbong Jang, Lourdes Agapito.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2109.01750)] [[Project](https://sites.google.com/view/wbjang/home/codenerf)] [[Github](https://github.com/wayne1123/code-nerf)]

* **NeRF-VAE: A Geometry Aware 3D Scene Generative Model.**<br>
*Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Soňa Mokrá, Danilo J. Rezende.*<br>
ICML 2021. [[PDF](https://arxiv.org/abs/2104.00587)]

**Pose-free**

* **Self-Calibrating Neural Radiance Fields.**<br>
*Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, Jaesik Park.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2108.13826)]

* **BARF: Bundle-Adjusting Neural Radiance Fields.**<br>
*[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/), [Wei-Chiu Ma](http://people.csail.mit.edu/weichium/), Antonio Torralba, Simon Lucey.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.06405)] [[Github](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)]

* **NeRF--: Neural Radiance Fields Without Known Camera Parameters.**<br>
*[Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en), [Shangzhe Wu](http://elliottwu.com), [Weidi Xie](https://weidixie.github.io/weidi-personal-webpage/), [Min Chen](https://sites.google.com/site/drminchen/home), [Victor Adrian Prisacariu](https://eng.ox.ac.uk/people/victor-prisacariu/).*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2102.07064)] [[Project](http://nerfmm.active.vision/)] [[Github](https://github.com/ActiveVisionLab/nerfmm)]

**Varying appearance**

* **NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections.**<br>
*[Ricardo Martin-Brualla](http://www.ricardomartinbrualla.com/), [Noha Radwan](https://scholar.google.com/citations?user=g98QcZUAAAAJ&hl=en), [Mehdi S. M. Sajjadi](https://research.google/people/105804/), [Jonathan T. Barron](https://jonbarron.info/), [Alexey Dosovitskiy](https://scholar.google.com/citations?user=FXNJRDoAAAAJ&hl=en), [Daniel Duckworth](http://www.stronglyconvex.com/about.html).*<br>
CVPR 2021 (oral). [[PDF](https://arxiv.org/abs/2008.02268)] [[Github](https://nerf-w.github.io/)]

* **NeRFReN: Neural Radiance Fields with Reflections.**<br>
*Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, Song-Hai Zhang.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.15234)] [[Project](https://bennyguo.github.io/nerfren/]

**Large-scale scene**

* **Block-NeRF: Scalable Large Scene Neural View Synthesis.**<br>
*Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, Henrik Kretzschmar.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2202.05263)] [[Project](https://waymo.com/research/block-nerf/)]

* **Urban Radiance Fields.**<br>
*[Konstantinos Rematas](http://www.krematas.com/), Andrew Liu, Pratul P. Srinivasan, Jonathan T. Barron, Andrea Tagliasacchi, Thomas Funkhouser, Vittorio Ferrari.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.14643)] [[Project](https://urban-radiance-fields.github.io/)]

* **Mega-NERF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs.**<br>
*Haithem Turki, Deva Ramanan, Mahadev Satyanarayanan.*<br>
CVPR 2022. [[PDF](https://openaccess.thecvf.com/content/CVPR2022/html/Turki_Mega-NERF_Scalable_Construction_of_Large-Scale_NeRFs_for_Virtual_Fly-Throughs_CVPR_2022_paper.html)]

* **BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering.**<br>
*Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, Dahua Lin.*<br>
ECCV 2022. [[PDF](https://arxiv.org/abs/2112.05504)] [[Project](https://city-super.github.io/citynerf)]

**Dynamic scene**

* **D-NeRF: Neural Radiance Fields for Dynamic Scenes.**<br>
*[Albert Pumarola](https://www.albertpumarola.com/), [Enric Corona](https://www.iri.upc.edu/people/ecorona/), [Gerard Pons-Moll](http://virtualhumans.mpi-inf.mpg.de/), [Francesc Moreno-Noguer](http://www.iri.upc.edu/people/fmoreno/).*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.13961)] [[Project](https://www.albertpumarola.com/research/D-NeRF/index.html)] [[Github](https://github.com/albertpumarola/D-NeRF)] [[Data](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0)]

* **Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction.**<br>
*Guy Gafni, Justus Thies, Michael Zollhöfer, Matthias Nießner.*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.03065)] [[Project](https://gafniguy.github.io/4D-Facial-Avatars/)] [[Video](https://youtu.be/m7oROLdQnjk)]

* **NSFF: Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes.**<br>
*[Zhengqi Li](https://www.cs.cornell.edu/~zl548/), [Simon Niklaus](https://sniklaus.com/welcome), [Noah Snavely](https://www.cs.cornell.edu/~snavely/), [Oliver Wang](https://research.adobe.com/person/oliver-wang/).*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.13084)] [[Project](http://www.cs.cornell.edu/~zl548/NSFF)] [[Github](https://github.com/zhengqili/Neural-Scene-Flow-Fields)]

* **Space-time Neural Irradiance Fields for Free-Viewpoint Video.**<br>
*[Wenqi Xian](https://www.cs.cornell.edu/~wenqixian/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Johannes Kopf](https://johanneskopf.de/), [Changil Kim](https://changilkim.com/).*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.12950)] [[Project](https://video-nerf.github.io/)]

* **Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video.**<br>
*Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, Christian Theobalt.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2012.12247)] [[Project](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/)] [[Github](https://github.com/facebookresearch/nonrigid_nerf)]

* **NeRFlow: Neural Radiance Flow for 4D View Synthesis and Video Processing.**<br>
*Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B. Tenenbaum, Jiajun Wu.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2012.09790)] [[Project](https://yilundu.github.io/nerflow/)]

* **Nerfies: Deformable Neural Radiance Fields.**<br>
*[Keunhong Park](https://keunhong.com/), [Utkarsh Sinha](https://utkarshsinha.com/), [Jonathan T. Barron](https://jonbarron.info/), [Sofien Bouaziz](http://sofienbouaziz.com/), [Dan B Goldman](https://www.danbgoldman.com/), [Steven M. Seitz](https://homes.cs.washington.edu/~seitz/), [Ricardo-Martin Brualla](http://www.ricardomartinbrualla.com/).*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2011.12948)] [[Project](https://nerfies.github.io/)] [[Github](https://github.com/google/nerfies)]

* **Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-time.**<br>
*[Liao Wang](https://aoliao12138.github.io/), [Jiakai Zhang](https://jiakai-zhang.github.io/), Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Lan Xu, Jingyi Yu.*<br>
CVPR 2022 (Oral). [[PDF](https://arxiv.org/abs/2202.08614)] [[Project](https://aoliao12138.github.io/FPO/)]

* **CoNeRF: Controllable Neural Radiance Fields.**<br>
*Kacper Kania, Kwang Moo Yi, Marek Kowalski, Tomasz Trzciński, Andrea Taliasacchi.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.01983)] [[Project](https://conerf.github.io/)]

## 3D Generative Models from Single Views

### Unconditional 3D Generative Models

* **HoloGAN: Unsupervised learning of 3D representations from natural images.**<br>
*[Thu Nguyen-Phuoc](https://monkeyoverflow.com/about/),  [Chuan Li](https://lambdalabs.com/blog/author/chuan/), Lucas Theis, [Christian Richardt]( https://richardt.name/), [Yong-liang Yang](http://yongliangyang.net/).*<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1904.01326)] [[Project](https://www.monkeyoverflow.com/hologan-unsupervised-learning-of-3d-representations-from-natural-images/] [[Github](https://github.com/thunguyenphuoc/HoloGAN)]

* **BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images.**<br>
*Thu Nguyen-Phuoc, Christian Richardt, Long Mai, Yong-Liang Yang, Niloy Mitra.*<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2002.08988)] [[Project](https://www.monkeyoverflow.com/#/blockgan/)] [[Github](https://github.com/thunguyenphuoc/BlockGAN)]

* **GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis.**<br>
*[Katja Schwarz](https://katjaschwarz.github.io/), [Yiyi Liao](https://yiyiliao.github.io/), [Michael Niemeyer](https://m-niemeyer.github.io/), [Andreas Geiger](http://www.cvlibs.net/).*<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2007.02442)] [[Project](https://avg.is.tuebingen.mpg.de/publications/schwarz2020neurips)] [[Github](https://github.com/autonomousvision/graf)]

* **pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis.**<br>
*[Eric R. Chan](https://ericryanchan.github.io/), [Marco Monteiro](https://marcoamonteiro.github.io/pi-GAN-website/), [Petr Kellnhofer](https://kellnhofer.xyz/), [Jiajun Wu](https://jiajunwu.com/), [Gordon Wetzstein](https://stanford.edu/~gordonwz/).*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.00926)] [[Project](https://marcoamonteiro.github.io/pi-GAN-website/)] [[Github](https://github.com/lucidrains/pi-GAN-pytorch)]

* **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields.**<br>
*Michael Niemeyer, Andreas Geiger.*<br>
CVPR 2021 (Best Paper). [[PDF](https://arxiv.org/abs/2011.12100)] [[Project](https://m-niemeyer.github.io/project-pages/giraffe/index.html)] [[Github](https://github.com/autonomousvision/giraffe)]

* **A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis.**<br>
*Xingang Pan, Xudong Xu, Chen Change Loy, Christian Theobalt, Bo Dai.*<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2110.15678)]

* **GRAM-HD: 3D-Consistent Image Generation at High Resolution with Generative Radiance Manifolds.**<br>
*Jianfeng Xiang, Jiaolong Yang, Yu Deng, Xin Tong.*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2206.07255)] [[Project](https://jeffreyxiang.github.io/GRAM-HD/)]

* **VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids.**<br>
*Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi Liao, Andreas Geiger.*<br>
arxiv 2022. [[PDF](https://arxiv.org/pdf/2206.07695.pdf)] [[Github](https://github.com/autonomousvision/voxgraf)]

* **CIPS-3D: A 3D-Aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis.**<br>
*Peng Zhou, Lingxi Xie, Bingbing Ni, Qi Tian.*<br>
arxiv 2021. [[PDF](https://arxiv.org/pdf/2110.09788.pdf)] [[Github](https://github.com/PeterouZh/CIPS-3D)]

* **EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks.**<br>
*[Eric R. Chan](https://ericryanchan.github.io/), [Connor Z. Lin](https://connorzlin.com/), [Matthew A. Chan](https://matthew-a-chan.github.io/), [Koki Nagano](https://luminohope.org/), [Boxiao Pan](https://cs.stanford.edu/~bxpan/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta), [Orazio Gallo](https://oraziogallo.github.io/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay), [Sameh Khamis](https://www.samehkhamis.com/), [Tero Karras](https://research.nvidia.com/person/tero-karras), [Gordon Wetzstein](https://stanford.edu/~gordonwz/).*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.07945)] [[Project](https://matthew-a-chan.github.io/EG3D)]

* **StylizedNeRF: Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D Mutual Learning.**<br>
*Yi-Hua Huang, Yue He, Yu-Jie Yuan, Yu-Kun Lai, Lin Gao.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2205.12183)]

* **Multi-View Consistent Generative Adversarial Networks for 3D-aware Image Synthesis.**<br>
*Xuanmeng Zhang, Zhedong Zheng, Daiheng Gao, Bang Zhang, Pan Pan, Yi Yang.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2204.06307)] [[Github](https://github.com/Xuanmeng-Zhang/MVCGAN)]

* **Disentangled3D: Learning a 3D Generative Model with Disentangled Geometry and Appearance from Monocular Images.**<br>
*[Ayush Tewari](https://ayushtewari.com/), Mallikarjun B R, Xingang Pan, Ohad Fried, Maneesh Agrawala, Christian Theobalt.*<br>
CVPR 2022. [[PDF](https://people.mpi-inf.mpg.de/~atewari/projects/D3D/data/paper.pdf)] [[Project](https://people.mpi-inf.mpg.de/~atewari/projects/D3D/)]

* **GIRAFFE HD: A High-Resolution 3D-aware Generative Model.**<br>
*Yang Xue, Yuheng Li, Krishna Kumar Singh, Yong Jae Lee.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2203.14954)]

* **StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation.**<br>
*[Roy Or-El](https://homes.cs.washington.edu/~royorel/), [Xuan Luo](https://roxanneluo.github.io/), Mengyi Shan, Eli Shechtman, Jeong Joon Park, Ira Kemelmacher-Shlizerman.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.11427)] [[Project](https://stylesdf.github.io/)] [[Github](https://github.com/royorel/StyleSDF)]

* **FENeRF: Face Editing in Neural Radiance Fields.**<br>
*Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, Jue Wang.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.15490)] 

* **LOLNeRF: Learn from One Look.**<br>
*[Daniel Rebain](https://vision.cs.ubc.ca/team/), Mark Matthews, Kwang Moo Yi, Dmitry Lagun, Andrea Tagliasacchi.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.09996)] [[Project](https://ubc-vision.github.io/lolnerf/)]

* **GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation.**<br>
*[Yu Deng](https://yudeng.github.io/), [Jiaolong Yang](https://jlyang.org/), [Jianfeng Xiang](http://www.xtong.info/), [Xin Tong]().*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.08867)] [[Project](https://yudeng.github.io/GRAM/)] [[Github](https://yudeng.github.io/GRAM/)]

* **3D-aware Image Synthesis via Learning Structural and Textural Representations.**<br>
*Yinghao Xu, Sida Peng, Ceyuan Yang, Yujun Shen, Bolei Zhou.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.10759)] [[Project](https://genforce.github.io/volumegan/)] [[Github](https://github.com/genforce/VolumeGAN)]

* **MOST-GAN: 3D Morphable StyleGAN for Disentangled Face Image Manipulation.**<br>
*Safa C. Medin, Bernhard Egger, Anoop Cherian, Ye Wang, Joshua B. Tenenbaum, Xiaoming Liu, Tim K. Marks.*<br>
AAAI 2022. [[PDF](https://arxiv.org/abs/2111.01048)]

* **Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks.**<br>
*Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, Jinwoo Shin.*<br>
ICLR 2022. [[PDF](https://openreview.net/forum?id=Czsdv-S4-w9)] [[Project](https://sihyun-yu.github.io/digan/)] [[Github](https://github.com/sihyun-yu/digan)]

* **StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis.**<br>
*[Jiatao Gu](http://jiataogu.me/), [Lingjie Liu](https://lingjie0206.github.io/), [Peng Wang](https://totoro97.github.io/about.html), [Christian Theobalt](http://people.mpi-inf.mpg.de/~theobalt/).*<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2110.08985)] [[Project](http://jiataogu.me/style_nerf/)]

* **EpiGRAF: Rethinking training of 3D GANs.**<br>
*[Ivan Skorokhodov](https://universome.github.io/), [Sergey Tulyakov](http://www.stulyakov.com/), [Yiqun Wang](https://sites.google.com/view/yiqun-wang/home), [Peter Wonka](https://peterwonka.net/).*<br>
NeurIPS 2022. [[PDF](https://arxiv.org/abs/2206.10535)] [[Project](https://universome.github.io/epigraf)] [[Github](https://github.com/universome/epigraf)]

* **Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis.**<br>
*Jeong-gi Kwak, Yuanming Li, Dongsik Yoon, Donghyeon Kim, David Han, Hanseok Ko.*<br>
ECCV 2022. [[PDF](https://arxiv.org/abs/2207.10257)] [[Project](https://jgkwak95.github.io/surfgan/)] [[Github](https://github.com/jgkwak95/SURF-GAN)]

* **Generative Multiplane Images: Making a 2D GAN 3D-Aware.**<br>
*[Xiaoming Zhao](https://xiaoming-zhao.com/), [Fangchang Ma](https://fangchangma.github.io/), [David Güera](https://scholar.google.com/citations?user=bckYvFkAAAAJ&hl=en), [Zhile Ren](https://jrenzhile.com/), [Alexander G. Schwing](https://www.alexander-schwing.de/), [Alex Colburn](https://www.colburn.org/).*<br>
ECCV 2022. [[PDF](https://arxiv.org/abs/2207.10642)] [[Project](https://xiaoming-zhao.github.io/projects/gmpi/)] [[Github](https://github.com/apple/ml-gmpi)]

* **3D-FM GAN: Towards 3D-Controllable Face Manipulation.**<br>
*[Yuchen Liu](https://lychenyoko.github.io/), Zhixin Shu, Yijun Li, Zhe Lin, Richard Zhang, and Sun-Yuan Kung.*<br>
ECCV 2022. [[PDF](https://arxiv.org/abs/2208.11257)] [[Project](https://lychenyoko.github.io/3D-FM-GAN-Webpage/)]

* **Improving 3D-aware Image Synthesis with A Geometry-aware Discriminator.**<br>
*Zifan Shi, Yinghao Xu, Yujun Shen, Deli Zhao, Qifeng Chen, Dit-Yan Yeung.*<br>
NeurIPS 2022. [[PDF](https://arxiv.org/abs/2209.15637)] [[Project](https://vivianszf.github.io/geod)]

* **VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids.**<br>
*Schwarz, Katja, Sauer, Axel, Niemeyer, Michael, Liao, Yiyi, and Geiger, Andreas.*<br>
NeurIPS 2022. [[PDF](https://arxiv.org/pdf/2206.07695.pdf)] [[Project](https://katjaschwarz.github.io/voxgraf)]

### Conditional 3D Generative Models

* **Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields.**<br>
*[Yuedong Chen](https://donydchen.github.io/), [Qianyi Wu](https://wuqianyi.top/), [Chuanxia Zheng](https://www.chuanxiaz.com/), [Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/), [Jianfei Cai](https://jianfei-cai.github.io/).*<br>
ECCV 2022. [[PDF](https://arxiv.org/abs/2203.10821)] [[Project](https://donydchen.github.io/sem2nerf)]

* **IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-aware Portrait Synthesis.**<br>
*[Jingxiang Sun](https://github.com/MrTornado24), [Xuan Wang](https://mrtornado24.github.io/IDE-3D/), [Yichun Shi](https://seasonsh.github.io/), [Lizhen Wang](https://lizhenwangt.github.io/), [Jue Wang](https://juewang725.github.io/), [Yebin Liu](https://liuyebin.com/).*<br>
SIGGRAPH Asia 2022. [[PDF](https://arxiv.org/abs/2205.15517)] [[Project](https://mrtornado24.github.io/IDE-3D/)] [[Github](https://github.com/MrTornado24/IDE-3D)]

* **GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds.**<br>
*Zekun Hao, Arun Mallya, Serge Belongie, Ming-Yu Liu.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.07659)] [[Project](https://nvlabs.github.io/GANcraft/)]

## 3D-aware Video Synthesis
* **Streaming Radiance Fields for 3D Video Synthesis.**<br>
*Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Ping Tan.*<br>
NeurIPS 2022. [[PDF](https://arxiv.org/abs/2210.14831)]

* **3D-Aware Video Generation.**<br>
*[Sherwin Bahmani](https://sherwinbahmani.github.io/), [Jeong Joon Park](https://jjparkcv.github.io/), [Despoina Paschalidou](https://paschalidoud.github.io/), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en/), [Gordon Wetzstein](https://stanford.edu/~gordonwz/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html/), [Radu Timofte](https://ee.ethz.ch/the-department/people-a-z/person-detail.MjAxNjc4.TGlzdC8zMjc5LC0xNjUwNTg5ODIw.html/).*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2206.14797)] [[Project](https://sherwinbahmani.github.io/3dvidgen/)] [[Github](https://github.com/sherwinbahmani/3dvideogeneration/)]

<img src="https://visitor-badge.glitch.me/badge?style=flat-square&amp;page_id=weihaox/awesome-3d-aware-synthesis" alt="visitors" />
<br>Unique visitors since Nov 2022

<div id="footer">
<div id="footer-text">
Page generated 2022-10-25, by <a href="http://jemdoc.jaboc.net/">jemdoc</a>.
</div>
</div>
