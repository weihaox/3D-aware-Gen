<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
<meta name="generator" content="jemdoc, see http://jemdoc.jaboc.net/" />
<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
<link rel="stylesheet" href="jemdoc.css" type="text/css" />
</head>

<!-- <div id="layout-content">
<div id="toptitle">
  <h1 align="center">3D-aware Image Synthesis &ndash; Papers, Codes and Datasets</h1>
</div> -->

<div id="layout-content">
<!-- <div id="toptitle"> -->
<p align="center">
  <h1 align="center">A Survey on Deep Generative 3D-aware Image Synthesis</h1> 
  <p align="center">
    ACM Computing Surveys, 2023
    <br />
    <a href="https://weihaox.github.io/"><strong>Weihao Xia</strong></a>
    ·
    <a href="http://www.homepages.ucl.ac.uk/~ucakjxu/"><strong>Jing-Hao Xue</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2210.14267'>
      <img src='https://img.shields.io/badge/Paper-Paper-green?style=flat&logo=arxiv&logoColor=green' alt='arxiv Paper'>
    </a>
    <a href='https://weihaox.github.io/3D-aware-Gen/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://dl.acm.org/doi/10.1145/3626193' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/CSUR-Paper-red?style=flat&logoColor=red' alt='CSUR Paper'>
    </a>
  </p>
</p>
<!-- </div> -->

Introduction
------------

This project lists representative papers/codes/datasets about deep **[3D-aware image synthesis](https://weihaox.github.io/3D-aware-Gen)**. Besides **3D-aware Generative Models** (GANs and Diffusion Models) discussed in this [survey](https://arxiv.org/abs/2210.14267), this project additionally covers novel view synthesis studies, especially those based on [implicit neural representations](https://github.com/weihaox/awesome-neural-rendering#implicit-neural-representation-and-rendering) such as NeRF. 

We aim to constantly update the latest relevant papers and help the community track this topic. Please feel free to join us and [contribute](https://github.com/weihaox/3D-aware-Gen/blob/main/CONTRIBUTING.md) to the project. Please do not hesitate to reach out if you have any questions or suggestions.

Survey paper
------------

*   [A Survey on Deep Generative 3D-aware Image Synthesis](https://arxiv.org/abs/2210.14267)  
    Weihao Xia and Jing-Hao Xue. _ACM Computing Surveys_, 2023. 

## 3D Control of 2D GANs

### 3D Control Latent Directions

For 3D control over diffusion models simiar to [GAN](https://github.com/weihaox/GAN-Inversion#gan-latent-space-editing), please refer to [semantic manipulation in diffusion latent spaces](https://github.com/weihaox/GAN-Inversion#semantic-editing-in-diffusion-latent-spaces).

* **SeFa: Closed-Form Factorization of Latent Semantics in GANs.**<br>
*Yujun Shen, Bolei Zhou.*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2007.06600)] [[Project](https://genforce.github.io/sefa/)] [[Code](https://github.com/genforce/sefa)] 

* **GANSpace: Discovering Interpretable GAN Controls.**<br>
*Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, Sylvain Paris.*<br>
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2004.02546)] [[Code](https://github.com/harskish/ganspace)]

* **Interpreting the Latent Space of GANs for Semantic Face Editing.**<br>
*[Yujun Shen](http://shenyujun.github.io/), [Jinjin Gu](http://www.jasongt.com/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml), [Bolei Zhou](http://bzhou.ie.cuhk.edu.hk/).*<br>
CVPR 2020. [[Paper](https://arxiv.org/abs/1907.10786)] [[Project](https://genforce.github.io/interfacegan/)] [[Code](https://github.com/genforce/interfacegan)]

* **Unsupervised Discovery of Interpretable Directions in the GAN Latent Space.**<br>
*Andrey Voynov, Artem Babenko.*<br>
ICML 2020. [[Paper](https://arxiv.org/abs/2002.03754)] [[Code](https://github.com/anvoynov/GANLatentDiscovery)]

* **On the "steerability" of generative adversarial networks.**<br>
*Ali Jahanian, Lucy Chai, Phillip Isola.*<br>
ICLR 2020. [[Paper](https://arxiv.org/abs/1907.07171)] [[Project](https://ali-design.github.io/gan_steerability/)] [[Code](https://github.com/ali-design/gan_steerability)]

### 3D Parameters as Controls

* **3D-FM GAN: Towards 3D-Controllable Face Manipulation.**<br>
*[Yuchen Liu](https://lychenyoko.github.io/), Zhixin Shu, Yijun Li, Zhe Lin, Richard Zhang, and Sun-Yuan Kung.*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2208.11257)] [[Project](https://lychenyoko.github.io/3D-FM-GAN-Webpage/)]

* **GAN-Control: Explicitly Controllable GANs.**<br>
*Alon Shoshan, Nadav Bhonker, Igor Kviatkovsky, Gerard Medioni.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2101.02477)] [[Project](https://alonshoshan10.github.io/gan_control/)] [[Code](https://github.com/amazon-science/gan-control)]

* **CONFIG: Controllable Neural Face Image Generation.**<br>
*Marek Kowalski, Stephan J. Garbin, Virginia Estellers, Tadas Baltrušaitis, Matthew Johnson, Jamie Shotton.*<br>
ECCV 2020. [[Paper](https://arxiv.org/abs/2005.02671)] [[Code](https://github.com/microsoft/ConfigNet)]

* **DiscoFaceGAN: Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning.**<br>
*Yu Deng, Jiaolong Yang, Dong Chen, Fang Wen, Xin Tong.*<br>
CVPR 2020. [[Paper](https://arxiv.org/Paper/2004.11660.Paper)] [[Code](https://github.com/microsoft/DiscoFaceGAN)] 

* **StyleRig: Rigging StyleGAN for 3D Control over Portrait Images.**<br>
*Ayush Tewari, Mohamed Elgharib, Gaurav Bharaj, Florian Bernard, Hans-Peter Seidel, Patrick Pérez, Michael Zollhöfer, Christian Theobalt.*<br>
CVPR 2020 (oral). [[Paper](https://arxiv.org/abs/2004.00121)] [[Project](https://gvv.mpi-inf.mpg.de/projects/StyleRig/)]

* **PIE: Portrait Image Embedding for Semantic Control.**<br> 
*[Ayush Tewari](http://people.mpi-inf.mpg.de/~atewari/), Mohamed Elgharib, Mallikarjun B R., Florian Bernard, Hans-Peter Seidel, Patrick Pérez, Michael Zollhöfer, Christian Theobalt.*<br> 
TOG (SIGGRAPH Asia) 2020. [[Paper](http://gvv.mpi-inf.mpg.de/projects/PIE/data/paper.Paper)] [[Project](http://gvv.mpi-inf.mpg.de/projects/PIE/)]

### 3D Prior Knowledge as Constraints

* **3D-Aware Indoor Scene Synthesis with Depth Priors.**<br>
*Zifan Shi, Yujun Shen, Jiapeng Zhu, Dit-Yan Yeung, Qifeng Chen.*<br>
ECCV 2022 (oral). [[Paper](https://arxiv.org/abs/2202.08553)] [[Project](https://vivianszf.github.io/depthgan/)] [[Code](https://github.com/vivianszf/depthgan)]

* **NGP: Towards a Neural Graphics Pipeline for Controllable Image Generation.**<br>
*Xuelin Chen, Daniel Cohen-Or, Baoquan Chen, Niloy J. Mitra.*<br>
Eurographics 2021. [[Paper](https://arxiv.org/abs/2006.10569)] [[Code](http://geometry.cs.ucl.ac.uk/projects/2021/ngp)]

* **Lifting 2D StyleGAN for 3D-Aware Face Generation.**<br>
*[Yichun Shi](https://seasonsh.github.io/), Divyansh Aggarwal, [Anil K. Jain](http://www.cse.msu.edu/~jain/).*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2011.13126)] [[Code](https://github.com/seasonSH/LiftedGAN)]

* **RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis.**<br>
*Atsuhiro Noguchi, Tatsuya Harada.*<br>
ICLR 2020. [[Paper](https://arxiv.org/abs/1909.12573)] [[Code](https://github.com/nogu-atsu/RGBD-GAN)]

* **Visual Object Networks: Image Generation with Disentangled 3D Representation.**<br>
*Jun-Yan Zhu, Zhoutong Zhang, Chengkai Zhang, Jiajun Wu, Antonio Torralba, Joshua B. Tenenbaum, William T. Freeman.*<br>
NeurIPS 2018. [[Paper](https://arxiv.org/abs/1812.02725)] [[Project](http://von.csail.mit.edu/)] [[Code](https://github.com/junyanz/VON)]

* **3D Shape Induction from 2D Views of Multiple Objects.**<br>
*Matheus Gadelha, Subhransu Maji, Rui Wang.*<br>
3DV 2017. [[Paper](https://arxiv.org/abs/1612.05872)] [[Project](http://mgadelha.me/prgan/index.html)] [[Code](https://github.com/matheusgadelha/PrGAN)]

* **Generative Image Modeling using Style and Structure Adversarial Networks.**<br>
*Xiaolong Wang, Abhinav Gupta.*<br>
ECCV 2016. [[Paper](https://arxiv.org/abs/1603.05631)] [[Project](https://github.com/facebook/eyescream)] [[Code](https://github.com/xiaolonw/ss-gan)]

## 3D-aware GANs for a Single Image Category

### Unconditional 3D Generative Models

* **BallGAN: 3D-aware Image Synthesis with a Spherical Background.**<br>
*Minjung Shin, Yunji Seo, Jeongmin Bae, Young Sun Choi, Hyunsu Kim, Hyeran Byun, Youngjung Uh.*<br>
ICCV 2023. [[Paper](https://arxiv.org/abs/2301.09091)] [[Project](https://minjung-s.github.io/ballgan/)] [[Code](https://github.com/minjung-s/BallGAN)]

* **Mimic3D: Thriving 3D-Aware GANs via 3D-to-2D Imitation.**<br>
*Xingyu Chen, Yu Deng, Baoyuan Wang.*<br>
ICCV 2023. [[Paper](https://arxiv.org/abs/2303.09036)] [[Project](https://seanchenxy.github.io/Mimic3DWeb/)]

* **GRAM-HD: 3D-Consistent Image Generation at High Resolution with Generative Radiance Manifolds.**<br>
*Jianfeng Xiang, Jiaolong Yang, Yu Deng, Xin Tong.*<br>
ICCV 2023. [[Paper](https://arxiv.org/abs/2206.07255)] [[Project](https://jeffreyxiang.github.io/GRAM-HD/)]

* **Live 3D Portrait: Real-Time Radiance Fields for Single-Image Portrait View Synthesis.**<br>
*Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, Koki Nagano.*<br>
TOG (SIGGRAPH) 2023. [[Paper](https://research.nvidia.com/labs/nxp/lp3d//media/paper.Paper)] [[Project](https://research.nvidia.com/labs/nxp/lp3d//)]

* **VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids.**<br>
*Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi Liao, Andreas Geiger.*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/Paper/2206.07695.Paper)] [[Code](https://github.com/autonomousvision/voxgraf)]

* **GeoD: Improving 3D-aware Image Synthesis with A Geometry-aware Discriminator.**<br>
*Zifan Shi, Yinghao Xu, Yujun Shen, Deli Zhao, Qifeng Chen, Dit-Yan Yeung.*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2209.15637)] [[Project](https://vivianszf.github.io/geod)]

* **EpiGRAF: Rethinking training of 3D GANs.**<br>
*[Ivan Skorokhodov](https://universome.github.io/), [Sergey Tulyakov](http://www.stulyakov.com/), [Yiqun Wang](https://sites.google.com/view/yiqun-wang/home), [Peter Wonka](https://peterwonka.net/).*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.10535)] [[Project](https://universome.github.io/epigraf)] [[Code](https://github.com/universome/epigraf)]

* **VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids.**<br>
*Schwarz, Katja, Sauer, Axel, Niemeyer, Michael, Liao, Yiyi, and Geiger, Andreas.*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/Paper/2206.07695.Paper)] [[Project](https://katjaschwarz.github.io/voxgraf)]

* **Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis.**<br>
*Jeong-gi Kwak, Yuanming Li, Dongsik Yoon, Donghyeon Kim, David Han, Hanseok Ko.*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2207.10257)] [[Project](https://jgkwak95.github.io/surfgan/)] [[Code](https://github.com/jgkwak95/SURF-GAN)]

* **Generative Multiplane Images: Making a 2D GAN 3D-Aware.**<br>
*[Xiaoming Zhao](https://xiaoming-zhao.com/), [Fangchang Ma](https://fangchangma.github.io/), [David Güera](https://scholar.google.com/citations?user=bckYvFkAAAAJ&hl=en), [Zhile Ren](https://jrenzhile.com/), [Alexander G. Schwing](https://www.alexander-schwing.de/), [Alex Colburn](https://www.colburn.org/).*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2207.10642)] [[Project](https://xiaoming-zhao.github.io/projects/gmpi/)] [[Code](https://github.com/apple/ml-gmpi)]

* **3D-FM GAN: Towards 3D-Controllable Face Manipulation.**<br>
*[Yuchen Liu](https://lychenyoko.github.io/), Zhixin Shu, Yijun Li, Zhe Lin, Richard Zhang, and Sun-Yuan Kung.*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2208.11257)] [[Project](https://lychenyoko.github.io/3D-FM-GAN-Webpage/)]

* **EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks.**<br>
*[Eric R. Chan](https://ericryanchan.github.io/), [Connor Z. Lin](https://connorzlin.com/), [Matthew A. Chan](https://matthew-a-chan.github.io/), [Koki Nagano](https://luminohope.org/), [Boxiao Pan](https://cs.stanford.edu/~bxpan/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta), [Orazio Gallo](https://oraziogallo.github.io/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay), [Sameh Khamis](https://www.samehkhamis.com/), [Tero Karras](https://research.nvidia.com/person/tero-karras), [Gordon Wetzstein](https://stanford.edu/~gordonwz/).*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.07945)] [[Project](https://matthew-a-chan.github.io/EG3D)] [[Code](https://github.com/NVlabs/eg3d)]

* **StylizedNeRF: Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D Mutual Learning.**<br>
*Yi-Hua Huang, Yue He, Yu-Jie Yuan, Yu-Kun Lai, Lin Gao.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2205.12183)]

* **Multi-View Consistent Generative Adversarial Networks for 3D-aware Image Synthesis.**<br>
*Xuanmeng Zhang, Zhedong Zheng, Daiheng Gao, Bang Zhang, Pan Pan, Yi Yang.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2204.06307)] [[Code](https://github.com/Xuanmeng-Zhang/MVCGAN)]

* **Disentangled3D: Learning a 3D Generative Model with Disentangled Geometry and Appearance from Monocular Images.**<br>
*[Ayush Tewari](https://ayushtewari.com/), Mallikarjun B R, Xingang Pan, Ohad Fried, Maneesh Agrawala, Christian Theobalt.*<br>
CVPR 2022. [[Paper](https://people.mpi-inf.mpg.de/~atewari/projects/D3D/data/paper.Paper)] [[Project](https://people.mpi-inf.mpg.de/~atewari/projects/D3D/)]

* **GIRAFFE HD: A High-Resolution 3D-aware Generative Model.**<br>
*Yang Xue, Yuheng Li, Krishna Kumar Singh, Yong Jae Lee.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2203.14954)] [[Code](https://github.com/AustinXY/GIRAFFEHD)]

* **StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation.**<br>
*[Roy Or-El](https://homes.cs.washington.edu/~royorel/), [Xuan Luo](https://roxanneluo.github.io/), Mengyi Shan, Eli Shechtman, Jeong Joon Park, Ira Kemelmacher-Shlizerman.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.11427)] [[Project](https://stylesdf.github.io/)] [[Code](https://github.com/royorel/StyleSDF)]

* **FENeRF: Face Editing in Neural Radiance Fields.**<br>
*Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, Jue Wang.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.15490)] [[Code](https://github.com/MrTornado24/FENeRF)]

* **LOLNeRF: Learn from One Look.**<br>
*[Daniel Rebain](https://vision.cs.ubc.ca/team/), Mark Matthews, Kwang Moo Yi, Dmitry Lagun, Andrea Tagliasacchi.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.09996)] [[Project](https://ubc-vision.github.io/lolnerf/)]

* **GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation.**<br>
*[Yu Deng](https://yudeng.github.io/), [Jiaolong Yang](https://jlyang.org/), [Jianfeng Xiang](http://www.xtong.info/), [Xin Tong]().*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.08867)] [[Project](https://yudeng.github.io/GRAM/)] [[Code](https://yudeng.github.io/GRAM/)]

* **VolumeGAN: 3D-aware Image Synthesis via Learning Structural and Textural Representations.**<br>
*Yinghao Xu, Sida Peng, Ceyuan Yang, Yujun Shen, Bolei Zhou.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.10759)] [[Project](https://genforce.github.io/volumegan/)] [[Code](https://github.com/genforce/VolumeGAN)]

* **Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks.**<br>
*Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, Jinwoo Shin.*<br>
ICLR 2022. [[Paper](https://openreview.net/forum?id=Czsdv-S4-w9)] [[Project](https://sihyun-yu.github.io/digan/)] [[Code](https://github.com/sihyun-yu/digan)]

* **StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis.**<br>
*[Jiatao Gu](http://jiataogu.me/), [Lingjie Liu](https://lingjie0206.github.io/), [Peng Wang](https://totoro97.github.io/about.html), [Christian Theobalt](http://people.mpi-inf.mpg.de/~theobalt/).*<br>
ICLR 2022. [[Paper](https://arxiv.org/abs/2110.08985)] [[Project](http://jiataogu.me/style_nerf/)]

* **MOST-GAN: 3D Morphable StyleGAN for Disentangled Face Image Manipulation.**<br>
*Safa C. Medin, Bernhard Egger, Anoop Cherian, Ye Wang, Joshua B. Tenenbaum, Xiaoming Liu, Tim K. Marks.*<br>
AAAI 2022. [[Paper](https://arxiv.org/abs/2111.01048)]

* **A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis.**<br>
*Xingang Pan, Xudong Xu, Chen Change Loy, Christian Theobalt, Bo Dai.*<br>
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2110.15678)]

* **pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis.**<br>
*[Eric R. Chan](https://ericryanchan.github.io/), [Marco Monteiro](https://marcoamonteiro.github.io/pi-GAN-website/), [Petr Kellnhofer](https://kellnhofer.xyz/), [Jiajun Wu](https://jiajunwu.com/), [Gordon Wetzstein](https://stanford.edu/~gordonwz/).*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2012.00926)] [[Project](https://marcoamonteiro.github.io/pi-GAN-website/)] [[Code](https://github.com/lucidrains/pi-GAN-pytorch)]

* **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields.**<br>
*Michael Niemeyer, Andreas Geiger.*<br>
CVPR 2021 (Best Paper). [[Paper](https://arxiv.org/abs/2011.12100)] [[Project](https://m-niemeyer.github.io/project-pages/giraffe/index.html)] [[Code](https://github.com/autonomousvision/giraffe)]

* **BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images.**<br>
*Thu Nguyen-Phuoc, Christian Richardt, Long Mai, Yong-Liang Yang, Niloy Mitra.*<br>
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2002.08988)] [[Project](https://www.monkeyoverflow.com/#/blockgan/)] [[Code](https://github.com/thunguyenphuoc/BlockGAN)]

* **GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis.**<br>
*[Katja Schwarz](https://katjaschwarz.github.io/), [Yiyi Liao](https://yiyiliao.github.io/), [Michael Niemeyer](https://m-niemeyer.github.io/), [Andreas Geiger](http://www.cvlibs.net/).*<br>
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2007.02442)] [[Project](https://avg.is.tuebingen.mpg.de/publications/schwarz2020neurips)] [[Code](https://github.com/autonomousvision/graf)]

* **HoloGAN: Unsupervised learning of 3D representations from natural images.**<br>
*[Thu Nguyen-Phuoc](https://monkeyoverflow.com/about/),  [Chuan Li](https://lambdalabs.com/blog/author/chuan/), Lucas Theis, [Christian Richardt]( https://richardt.name/), [Yong-liang Yang](http://yongliangyang.net/).*<br>
ICCV 2019. [[Paper](https://arxiv.org/abs/1904.01326)] [[Project](https://www.monkeyoverflow.com/hologan-unsupervised-learning-of-3d-representations-from-natural-images/)] [[Code](https://github.com/thunguyenphuoc/HoloGAN)]

### Conditional 3D Generative Models

* **Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields.**<br>
*[Yuedong Chen](https://donydchen.github.io/), [Qianyi Wu](https://wuqianyi.top/), [Chuanxia Zheng](https://www.chuanxiaz.com/), [Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/), [Jianfei Cai](https://jianfei-cai.github.io/).*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2203.10821)] [[Project](https://donydchen.github.io/sem2nerf)] [[Code](https://github.com/donydchen/sem2nerf)]

* **IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-aware Portrait Synthesis.**<br>
*[Jingxiang Sun](https://github.com/MrTornado24), [Xuan Wang](https://mrtornado24.github.io/IDE-3D/), [Yichun Shi](https://seasonsh.github.io/), [Lizhen Wang](https://lizhenwangt.github.io/), [Jue Wang](https://juewang725.github.io/), [Yebin Liu](https://liuyebin.com/).*<br>
SIGGRAPH Asia 2022. [[Paper](https://arxiv.org/abs/2205.15517)] [[Project](https://mrtornado24.github.io/IDE-3D/)] [[Code](https://github.com/MrTornado24/IDE-3D)]

* **NeRFFaceEditing: Disentangled Face Editing in Neural Radiance Fields.**<br>
*Kaiwen Jiang, [Shu-Yu Chen](http://people.geometrylearning.com/csy/), [Feng-Lin Liu](http://people.geometrylearning.com/lfl/), [Hongbo Fu](http://sweb.cityu.edu.hk/hongbofu/), [Lin Gao](http://www.geometrylearning.com/cn/).*<br>
SIGGRAPH Asia 2022. [[Paper](https://arxiv.org/abs/2211.07968)] [[Project](http://geometrylearning.com/NeRFFaceEditing/)] 

* **GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds.**<br>
*Zekun Hao, Arun Mallya, Serge Belongie, Ming-Yu Liu.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2104.07659)] [[Project](https://nvlabs.github.io/GANcraft/)] [[Code](https://github.com/NVlabs/imaginaire)]

## 3D-aware Diffusion Models for a Single Image Category

* **Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction.**<br>
*Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, Hao Su.*<br> 
ICCV 2023. [[PDF](http://arxiv.org/abs/2304.06714)] [[Project](https://lakonik.github.io/ssdnerf)] [[Code](https://github.com/Lakonik/SSDNeRF)]

* **3D-aware Image Generation using 2D Diffusion Models.**<br>
*[Jianfeng Xiang](https://jeffreyxiang.github.io/), Jiaolong Yang, Binbin Huang, Xin Tong.*<br> 
ICCV 2023. [[Paper](https://arxiv.org/abs/2303.17905)] [[Project](https://jeffreyxiang.github.io/ivid/)] [[Code](https://github.com/JeffreyXiang/ivid)]

* **HoloFusion: Towards Photo-realistic 3D Generative Modeling.**<br>
*Animesh Karnewar, Niloy J. Mitra, Andrea Vedaldi, David Novotny.*<br> 
ICCV 2023. [[Paper](http://arxiv.org/abs/2308.14244)] [[Project](https://holodiffusion.github.io/holofusion)]

* **HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion.**<br>
*[Ziya Erkoç](https://ziyaerkoc.com/), [Fangchang Ma](https://fangchangma.github.io/), [Qi Shan](http://shanqi.github.io/), [Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html), [Angela Dai](https://www.3dunderstanding.org/team.html).*<br>
ICCV 2023. [[Paper](https://arxiv.org/abs/2303.17015)] [[Project](https://ziyaerkoc.com/hyperdiffusion/)] 

* **LatentSwap3D: Semantic Edits on 3D Image GANs.**<br>
*Enis Simsar, Alessio Tonioni, Evin Pınar Örnek, Federico Tombari.*<br> 
ICCV 2023 Workshop on AI3DCC. [[Paper](https://arxiv.org/abs/2212.01381)]

* **DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising Diffusion Models.**<br>
*[Jamie Wynn](https://scholar.google.com/citations?user=ASP-uu4AAAAJ&hl=en&oi=ao) and [Daniyar Turmukhambetov](https://scholar.google.com/citations?user=ELFm0CgAAAAJ&hl=en&oi=ao).*<br>
CVPR 2023. [[Paper](https://arxiv.org/abs/2302.12231)] [[Supplementary material](https://storage.googleapis.com/niantic-lon-static/research/diffusionerf/diffusionerf_supplemental.Paper)] [[COde](https://github.com/nianticlabs/diffusionerf)]

* **NeuralField-LDM: Scene Generation with Hierarchical Latent Diffusion Models.**<br>
*Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, Sanja Fidler.*<br>
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.09787)] [[Project](https://research.nvidia.com/labs/toronto-ai/NFLDM/)]

* **Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion.**<br>
*Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, Baining Guo.*<br>
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.06135)] [[Project](https://3d-avatar-diffusion.microsoft.com/)]

* **DiffRF: Rendering-guided 3D Radiance Field Diffusion.**<br>
*[Norman Müller](https://niessnerlab.org/members/norman_mueller/profile.html), [Yawar Siddiqui](https://niessnerlab.org/members/yawar_siddiqui/profile.html), [Lorenzo Porzi](https://scholar.google.com/citations?user=vW1gaVEAAAAJ), [Samuel Rota Bulò](https://scholar.google.com/citations?hl=de&user=484sccEAAAAJ), [Peter Kontschieder](https://scholar.google.com/citations?user=CxbDDRMAAAAJ&hl=en), [Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html).*<br> 
CVPR 2023 (Highlight). [[Paper](https://arxiv.org/abs/2212.01206)] [[Project](https://sirwyver.github.io/DiffRF/)]

* **RenderDiffusion: Image Diffusion for 3D Reconstruction, Inpainting and Generation.**<br>
*Titas Anciukevičius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J. Mitra, Paul Guerrero.*<br> 
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.09869)] [[Project](https://holodiffusion.github.io/)] [[Code](https://github.com/Anciukevicius/RenderDiffusion)]

* **SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction.**<br>
*[Zhizhuo Zhou](https://www.zhiz.dev/), [Shubham Tulsiani](https://shubhtuls.github.io/).*<br> 
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.00792)] [[Project](https://sparsefusion.github.io/)] [[Code](https://github.com/zhizdev/sparsefusion)]

* **HoloDiffusion: Training a 3D Diffusion Model using 2D Images.**<br>
*Animesh Karnewar, Andrea Vedaldi, David Novotny, Niloy Mitra.*<br> 
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.16509)] [[Project](https://3d-diffusion.github.io/)]

* **3DiM: Novel View Synthesis with Diffusion Models.**<br>
*Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, Mohammad Norouzi.*<br> 
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.04628)] [[Project](https://3d-diffusion.github.io/)]

* **3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models.**<br>
*[Biao Zhang](https://1zb.github.io/), [Jiapeng Tang](https://tangjiapeng.github.io/), [Matthias Niessner](https://www.niessnerlab.org/), [Peter Wonka](http://peterwonka.net/).*<br>
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2301.11445)] [[Project](https://1zb.github.io/3DShape2VecSet/)] [[Code](https://github.com/1zb/3DShape2VecSet)]

* **GAUDI: A Neural Architect for Immersive 3D Scene Generation.**<br>
*Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, Afshin Dehghan, Josh Susskind.*<br> 
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2212.01381)] [[Project](https://github.com/apple/ml-gaudi)]

* **Learning a Diffusion Prior for NeRFs.**<br>
*Guandao Yang, Abhijit Kundu, Leonidas J. Guibas, Jonathan T. Barron, Ben Poole.*<br> 
arxiv 2023. [[Paper](https://arxiv.org/abs/2304.14473)] 

* **Adding 3D Geometry Control to Diffusion Models.**<br>
*[Wufei Ma](https://wufeim.github.io/), [Qihao Liu](https://qihao067.github.io/), [Jiahao Wang](https://jiahaoplus.github.io/), Angtian Wang, [Yaoyao Liu](https://www.cs.jhu.edu/~yyliu/), [Adam Kortylewski](https://adamkortylewski.com/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/).*<br>
arxiv 2023. [[Paper](https://arxiv.org/abs/2306.08103)]

* **Generative Novel View Synthesis with 3D-Aware Diffusion Models.**<br>
*Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, Gordon Wetzstein.*<br> 
arxiv 2023. [[Paper](https://arxiv.org/abs/2304.02602)] [[Project](https://nvlabs.github.io/genvs/)] [[Code](https://github.com/NVlabs/genvs)]

* **3D-LDM: Neural Implicit 3D Shape Generation with Latent Diffusion Models.**<br>
*Gimin Nam, Mariem Khlifi, Andrew Rodriguez, Alberto Tono, Linqi Zhou, Paul Guerrero.*<br> 
arxiv 2022. [[Paper](https://arxiv.org/abs/2212.00842)]

## 3D-Aware Generative Models on ImageNet

* **VQ3D: Learning a 3D-Aware Generative Model on ImageNet.**<br>
*Kyle Sargent, Jing Yu Koh, Han Zhang, Huiwen Chang, Charles Herrmann, Pratul Srinivasan, Jiajun Wu, Deqing Sun.*<br>
ICCV 2023 (Oral). [[Paper](https://arxiv.org/abs/2302.06833)] [[Project](http://kylesargent.github.io/vq3d)]

* **3D Generation on ImageNet.**<br>
*Ivan Skorokhodov, Aliaksandr Siarohin, Yinghao Xu, Jian Ren, Hsin-Ying Lee, Peter Wonka, Sergey Tulyakov.*<br>
ICLR 2023 (Oral). [[Paper](https://openreview.net/forum?id=U2WjB9xxZ9q)] [[Project](https://u2wjb9xxz9q.github.io/)] [[Code](https://justimyhxu.github.io/pub.html)]

## 3D-aware Video Synthesis

* **3D-Aware Video Generation.**<br>
*[Sherwin Bahmani](https://sherwinbahmani.github.io/), [Jeong Joon Park](https://jjparkcv.github.io/), [Despoina Paschalidou](https://paschalidoud.github.io/), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en/), [Gordon Wetzstein](https://stanford.edu/~gordonwz/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html/), [Radu Timofte](https://ee.ethz.ch/the-department/people-a-z/person-detail.MjAxNjc4.TGlzdC8zMjc5LC0xNjUwNTg5ODIw.html/).*<br>
TMLR 2023. [[Paper](https://arxiv.org/abs/2206.14797)] [[Project](https://sherwinbahmani.github.io/3dvidgen/)] [[Code](https://github.com/sherwinbahmani/3dvideogeneration/)]

* **Streaming Radiance Fields for 3D Video Synthesis.**<br>
*Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Ping Tan.*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2210.14831)] [[Code](https://github.com/AlgoHunt/StreamRF)]

## INR-based 3D Novel View Synthesis

### Neural Scene Representations

* **Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations.**<br>
*Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas Funkhouser, Andrea Tagliasacchi.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.13152)] [[Project](https://srt-paper.github.io/)] [[Code](https://github.com/stelzner/srt)]

* **Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering.**<br>
*Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, Fredo Durand.*<br>
NeurIPS 2021 (Spotlight). [[Paper](https://arxiv.org/abs/2106.02634)] [[Project](https://vsitzmann.github.io/lfns/)] [[Code](https://github.com/vsitzmann/light-field-networks)]

* **Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields.**<br>
*[Jonathan T. Barron](https://jonbarron.info/), [Ben Mildenhall](https://bmild.github.io/), [Matthew Tancik](https://www.matthewtancik.com/), [Peter Hedman](https://phogzone.com/cv.html), [Ricardo Martin-Brualla](http://ricardomartinbrualla.com/), [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/).*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2103.13415)] [[Project](http://jonbarron.info/mipnerf)] [[Github](https://github.com/google/mipnerf)]

* **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.**<br>
*[Ben Mildenhall](http://people.eecs.berkeley.edu/~bmild/), [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/), [Matthew Tancik](http://www.matthewtancik.com/), [Jonathan T. Barron](https://jonbarron.info/), [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/), [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html).*<br>
ECCV 2020. [[Paper](https://arxiv.org/abs/2003.08934)] [[Project](http://tancik.com/nerf)] [[Gtihub-Tensorflow](https://github.com/bmild/nerf)] [[krrish94-PyTorch](https://github.com/krrish94/nerf-pytorch)] [[yenchenlin-PyTorch](https://github.com/yenchenlin/nerf-pytorch)]

* **Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision.**<br>
*Michael Niemeyer, Lars Mescheder, Michael Oechsle, Andreas Geiger.*<br>
CVPR 2020. [[Paper](http://www.cvlibs.net/publications/Niemeyer2020CVPR.Paper)] [[Code](https://github.com/autonomousvision/differentiable_volumetric_rendering)]

* **Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations.**<br>
*[Vincent Sitzmann](https://vsitzmann.github.io/), Michael Zollhöfer, Gordon Wetzstein.*<br>
NeurIPS 2019 (Oral, Honorable Mention "Outstanding New Directions").
[[Paper](http://arxiv.org/abs/1906.01618)] [[Project](https://github.com/vsitzmann/scene-representation-networks)] [[Code](https://github.com/vsitzmann/scene-representation-networks)] [[Dataset](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90?usp=sharing)]

* **LLFF: Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines.**<br>
*[Ben Mildenhall](http://people.eecs.berkeley.edu/~bmild/), Pratul Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, Abhishek Kar.*<br>
SIGGRAPH 2019. [[Paper](https://arxiv.org/abs/1905.00889)] [[Project](https://people.eecs.berkeley.edu/~bmild/llff/)] [[Code](https://github.com/Fyusion/LLFF)]

* **DeepVoxels: Learning Persistent 3D Feature Embeddings.**<br>
*Vincent Sitzmann, Justus Thies, Felix Heide, Matthias Nießner, Gordon Wetzstein, Michael Zollhöfer.*<br>
CVPR 2019 (Oral). [[Paper](https://arxiv.org/abs/1812.01024)] [[Project](http://vsitzmann.github.io/deepvoxels/)] [[Code](https://github.com/vsitzmann/deepvoxels)]

### Acceleration

* **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.**<br>
*[Thomas Müller](https://tom94.net/), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller).*<br>
SIGGRAPH (TOG) 2022. [[Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.Paper)] [[Project](https://nvlabs.github.io/instant-ngp)] [[Code](https://github.com/NVlabs/instant-ngp)]

* **DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering.**<br>
*[Liwen Wu](https://lwwu2.github.io/), [Jae Yong Lee](https://jyl.kr/), [Anand Bhattad](https://anandbhattad.github.io/), [Yuxiong Wang](https://yxw.web.illinois.edu/), [David A. Forsyth](http://luthuli.cs.uiuc.edu/~daf/).*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.10427)] [[Project](https://lwwu2.github.io/diver/)] [[Code](https://github.com/lwwu2/diver)]

* **KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs.**<br>
*Christian Reiser, Songyou Peng, Yiyi Liao, Andreas Geiger.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2103.13744)] [[Code](https://github.com/creiser/kilonerf)]

* **FastNeRF: High-Fidelity Neural Rendering at 200FPS.**<br>
*Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2103.10380)]

* **PlenOctrees for Real-time Rendering of Neural Radiance Fields.**<br>
*[Alex Yu](https://alexyu.net/), [Ruilong Li](https://www.liruilong.cn/), [Matthew Tancik](https://www.matthewtancik.com/), [Hao Li](https://www.hao-li.com/), [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/).*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2103.14024)] [[Project](https://alexyu.net/plenoctrees/)] [[Code](https://github.com/sxyu/plenoctree)]

* **Baking Neural Radiance Fields for Real-Time View Synthesis.**<br>
*[Peter Hedman](https://phogzone.com/), [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/), [Ben Mildenhall](https://bmild.github.io/), [Jonathan T. Barron](https://jonbarron.info/), [Paul Debevec](https://www.pauldebevec.com/).*<br>
ICCV 2021 (oral). [[Paper](https://arxiv.org/abs/2103.14645)] [[Project](https://nerf.live/)] [[Code](https://github.com/google-research/google-research/tree/master/snerg)]

* **AutoInt: Automatic Integration for Fast Neural Volume Rendering.**<br>
*David B. Lindell, Julien N. P. Martel, Gordon Wetzstein.*<br>
CVPR 2021 (oral). [[Paper](https://arxiv.org/abs/2012.01714)] [[Project](http://www.computationalimaging.org/publications/automatic-integration/)] [[Code](https://github.com/computational-imaging/automatic-integration)]

* **NSVF: Neural Sparse Voxel Fields.**<br>
*[Lingjie Liu](https://lingjie0206.github.io/), Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, Christian Theobalt.*<br>
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2007.11571)] [[Project](https://lingjie0206.github.io/papers/NSVF/)] [[Code](https://github.com/facebookresearch/NSVF)]

### From Constrained to In-the-wild Conditions

#### Few Images

* **GRF: Learning a General Radiance Field for 3D Representation and Rendering.**<br>
*Alex Trevithick, Bo Yang.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Trevithick_GRF_Learning_a_General_Radiance_Field_for_3D_Representation_and_ICCV_2021_paper.html)] [[Code](https://github.com/alextrevithick/GRF)]

* **MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo.**<br>
*[Anpei Chen](https://apchenstu.github.io/), [Zexiang Xu](http://cseweb.ucsd.edu/~zex014/), Fuqiang Zhao, Xiaoshuai Zhang, [Fanbo Xiang](https://www.fbxiang.com/), [Jingyi Yu](http://vic.shanghaitech.edu.cn/vrvc/en/people/), [Hao Su](https://cseweb.ucsd.edu/~haosu/).*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2103.15595)] [[Project](https://apchenstu.github.io/mvsnerf/)] [[Code](https://github.com/apchenstu/mvsnerf)]

* **CodeNeRF: Disentangled Neural Radiance Fields for Object Categories.**<br>
*Wonbong Jang, Lourdes Agapito.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2109.01750)] [[Project](https://sites.google.com/view/wbjang/home/codenerf)] [[Code](https://github.com/wayne1123/code-nerf)]

* **pixelNeRF: Neural Radiance Fields from One or Few Images.**<br>
*[Alex Yu](https://alexyu.net/), Vickie Ye, Matthew Tancik, Angjoo Kanazawa.*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2012.02190)] [[Project](https://alexyu.net/pixelnerf)] [[Code](https://github.com/sxyu/pixel-nerf)]

* **IBRNet: Learning Multi-View Image-Based Rendering.**<br>
*Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, Thomas Funkhouser.*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2102.13090)] [[Project](https://ibrnet.github.io/)] [[Code](https://github.com/googleinterns/IBRNet)]

* **NeRF-VAE: A Geometry Aware 3D Scene Generative Model.**<br>
*Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Soňa Mokrá, Danilo J. Rezende.*<br>
ICML 2021. [[Paper](https://arxiv.org/abs/2104.00587)]

#### Pose-free

* **Self-Calibrating Neural Radiance Fields.**<br>
*Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, Jaesik Park.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2108.13826)] [[Project](https://postech-cvlab.github.io/SCNeRF/)] [[Code](https://github.com/POSTECH-CVLab/SCNeRF)]

* **BARF: Bundle-Adjusting Neural Radiance Fields.**<br>
*[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/), [Wei-Chiu Ma](http://people.csail.mit.edu/weichium/), Antonio Torralba, Simon Lucey.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2104.06405)] [[Code](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)]

* **NeRF--: Neural Radiance Fields Without Known Camera Parameters.**<br>
*[Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en), [Shangzhe Wu](http://elliottwu.com), [Weidi Xie](https://weidixie.github.io/weidi-personal-webpage/), [Min Chen](https://sites.google.com/site/drminchen/home), [Victor Adrian Prisacariu](https://eng.ox.ac.uk/people/victor-prisacariu/).*<br>
arxiv 2021. [[Paper](https://arxiv.org/abs/2102.07064)] [[Project](http://nerfmm.active.vision/)] [[Code](https://github.com/ActiveVisionLab/nerfmm)]

#### Varying Appearance

* **NeRFReN: Neural Radiance Fields with Reflections.**<br>
*Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, Song-Hai Zhang.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.15234)] [[Project](https://bennyguo.github.io/nerfren/]

* **NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections.**<br>
*[Ricardo Martin-Brualla](http://www.ricardomartinbrualla.com/), [Noha Radwan](https://scholar.google.com/citations?user=g98QcZUAAAAJ&hl=en), [Mehdi S. M. Sajjadi](https://research.google/people/105804/), [Jonathan T. Barron](https://jonbarron.info/), [Alexey Dosovitskiy](https://scholar.google.com/citations?user=FXNJRDoAAAAJ&hl=en), [Daniel Duckworth](http://www.stronglyconvex.com/about.html).*<br>
CVPR 2021 (oral). [[Paper](https://arxiv.org/abs/2008.02268)] [[Code](https://nerf-w.github.io/)]

#### Large-scale Scene

* **Grid-guided Neural Radiance Fields for Large Urban Scenes.**<br>
*Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan, Nanxuan Zhao, Christian Theobalt, Bo Dai, Dahua Lin.*<br> 
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.14001)] [[Project](https://city-super.github.io/gridnerf/)]

* **S3-NeRF: Neural Reflectance Field from Shading and Shadow under a Single Viewpoint.**<br>
*[Wenqi Yang](https://ywq.github.io/), [Guanying Chen](https://guanyingc.github.io/), [Chaofeng Chen](http://chaofengc.github.io/), [Zhenfang Chen](https://zfchenunique.github.io/), [Kwan-Yee K. Wong](http://i.cs.hku.hk/~kykwong/).*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2210.08936)] [[Project](https://ywq.github.io/s3nerf)]

* **BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering.**<br>
*Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, Dahua Lin.*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2112.05504)] [[Project](https://city-super.github.io/citynerf)]

* **Block-NeRF: Scalable Large Scene Neural View Synthesis.**<br>
*Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, Henrik Kretzschmar.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2202.05263)] [[Project](https://waymo.com/research/block-nerf/)]

* **Urban Radiance Fields.**<br>
*[Konstantinos Rematas](http://www.krematas.com/), Andrew Liu, Pratul P. Srinivasan, Jonathan T. Barron, Andrea Tagliasacchi, Thomas Funkhouser, Vittorio Ferrari.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14643)] [[Project](https://urban-radiance-fields.github.io/)]

* **Mega-NERF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs.**<br>
*Haithem Turki, Deva Ramanan, Mahadev Satyanarayanan.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Turki_Mega-NERF_Scalable_Construction_of_Large-Scale_NeRFs_for_Virtual_Fly-Throughs_CVPR_2022_paper.html)] [[Code](https://github.com/cmusatyalab/mega-nerf)]

* **Shadow Neural Radiance Fields for Multi-view Satellite Photogrammetry.**<br>
*Dawa Derksen, Dario Izzo.*<br> 
CVPR 2021. [[Paper](https://arxiv.org/abs/2104.09877)] [[Code](https://github.com/esa/snerf)]

#### Dynamic Scene

* **NeRFPlayer: A Streamable Dynamic Scene Representation with Decomposed Neural Radiance Fields.**<br>
*Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, Andreas Geiger.*<br>
TVCG 2023. [[Paper](https://arxiv.org/abs/2210.15947)] [[Project](https://lsongx.github.io/projects/nerfplayer.html)]

* **Generative Deformable Radiance Fields for Disentangled Image Synthesis of Topology-Varying Objects.**<br>
*Ziyu Wang, Yu Deng, Jiaolong Yang, Jingyi Yu, Xin Tong.*<br>
Pacific Graphics 2022. [[Paper](https://arxiv.org/abs/2209.04183)] [[Code](https://ziyuwang98.github.io/GDRF/)]

* **Neural Surface Reconstruction of Dynamic Scenes with Monocular RGB-D Camera.**<br>
*[Hongrui Cai](https://rainbowrui.github.io/), [Wanquan Feng](https://github.com/WanquanF), [Xuetao Feng](https://scholar.google.com/citations?hl=en&user=5G-2EFcAAAAJ), [Yan Wang](), [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/).*<br>
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.15258)] [[Project](https://ustc3dv.github.io/ndr/)] [[Code](https://github.com/USTC3DV/NDR-code)]

* **LoRD: Local 4D Implicit Representation for High-Fidelity Dynamic Human Modeling.**<br>
*Boyan Jiang, Xinlin Ren, Mingsong Dou, Xiangyang Xue, Yanwei Fu, Yinda Zhang.*<br>
ECCV 2022. [[Paper](https://arxiv.org/abs/2208.08622)] [[Code](https://boyanjiang.github.io/LoRD/)]

* **Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-time.**<br>
*[Liao Wang](https://aoliao12138.github.io/), [Jiakai Zhang](https://jiakai-zhang.github.io/), Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Lan Xu, Jingyi Yu.*<br>
CVPR 2022 (Oral). [[Paper](https://arxiv.org/abs/2202.08614)] [[Project](https://aoliao12138.github.io/FPO/)]

* **CoNeRF: Controllable Neural Radiance Fields.**<br>
*Kacper Kania, Kwang Moo Yi, Marek Kowalski, Tomasz Trzciński, Andrea Taliasacchi.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.01983)] [[Project](https://conerf.github.io/)]

* **Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video.**<br>
*Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, Christian Theobalt.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2012.12247)] [[Project](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/)] [[Code](https://github.com/facebookresearch/nonrigid_nerf)]

* **NeRFlow: Neural Radiance Flow for 4D View Synthesis and Video Processing.**<br>
*Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B. Tenenbaum, Jiajun Wu.*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2012.09790)] [[Project](https://yilundu.github.io/nerflow/)]

* **Nerfies: Deformable Neural Radiance Fields.**<br>
*[Keunhong Park](https://keunhong.com/), [Utkarsh Sinha](https://utkarshsinha.com/), [Jonathan T. Barron](https://jonbarron.info/), [Sofien Bouaziz](http://sofienbouaziz.com/), [Dan B Goldman](https://www.danbgoldman.com/), [Steven M. Seitz](https://homes.cs.washington.edu/~seitz/), [Ricardo-Martin Brualla](http://www.ricardomartinbrualla.com/).*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2011.12948)] [[Project](https://nerfies.github.io/)] [[Code](https://github.com/google/nerfies)]

* **D-NeRF: Neural Radiance Fields for Dynamic Scenes.**<br>
*[Albert Pumarola](https://www.albertpumarola.com/), [Enric Corona](https://www.iri.upc.edu/people/ecorona/), [Gerard Pons-Moll](http://virtualhumans.mpi-inf.mpg.de/), [Francesc Moreno-Noguer](http://www.iri.upc.edu/people/fmoreno/).*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2011.13961)] [[Project](https://www.albertpumarola.com/research/D-NeRF/index.html)] [[Code](https://github.com/albertpumarola/D-NeRF)] [[Data](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0)]

* **Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction.**<br>
*Guy Gafni, Justus Thies, Michael Zollhöfer, Matthias Nießner.*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2012.03065)] [[Project](https://gafniguy.github.io/4D-Facial-Avatars/)] [[Video](https://youtu.be/m7oROLdQnjk)]

* **NSFF: Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes.**<br>
*[Zhengqi Li](https://www.cs.cornell.edu/~zl548/), [Simon Niklaus](https://sniklaus.com/welcome), [Noah Snavely](https://www.cs.cornell.edu/~snavely/), [Oliver Wang](https://research.adobe.com/person/oliver-wang/).*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2011.13084)] [[Project](http://www.cs.cornell.edu/~zl548/NSFF)] [[Code](https://github.com/zhengqili/Neural-Scene-Flow-Fields)]

* **Space-time Neural Irradiance Fields for Free-Viewpoint Video.**<br>
*[Wenqi Xian](https://www.cs.cornell.edu/~wenqixian/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Johannes Kopf](https://johanneskopf.de/), [Changil Kim](https://changilkim.com/).*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2011.12950)] [[Project](https://video-nerf.github.io/)]

The following papers are not directly related to 3D-aware image synthesis. But it would be beneficial to pay attention to those works. For example, in our survey, inverse rendering are not classified as 3D-aware image synthesis as they are not deliberately designed for this purpose. But with the inferred intrinsic components, photorealistic images can be rendered. 3D reconstruction models geometry only with no appearance information, meaning them not able to render images with photorealistic textures. But these representations have been introduced as the geometric representation along with a textural representation (e.g., Texture Field) for 3D-aware image synthesis.

## 3D Representations

* **K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.**<br>
*[Sara Fridovich-Keil](https://people.eecs.berkeley.edu/~sfk/), [Giacomo Meanti](https://www.iit.it/web/iit-mit-usa/people-details/-/people/giacomo-meanti), [Frederik Warburg](https://frederikwarburg.github.io/), [Benjamin Recht](https://people.eecs.berkeley.edu/~brecht/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/).*<br>
CVPR 2023. [[Paper](https://arxiv.org/abs/2301.10241)] [[Project](https://sarafridov.github.io/K-Planes/)] [[Code](https://github.com/sarafridov/K-Planes)]

* **HexPlane: A Fast Representation for Dynamic Scenes.**<br>
*[Ang Cao](https://caoang327.github.io/), [Justin Johnson](https://web.eecs.umich.edu/~justincj).*<br>
CVPR 2023. [[Paper](https://arxiv.org/abs/2301.09632)] [[Project](https://caoang327.github.io/HexPlane/)] [[Code](https://caoang327.github.io/HexPlane/)]

* **GIFS: Neural Implicit Function for General Shape Representation.**<br>
*Jianglong Ye, Yuntao Chen, Naiyan Wang, Xiaolong Wang.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2204.07126)] [[Project](https://jianglongye.com/gifs)] [[Code](https://github.com/jianglongye/gifs)]

* **Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields.**<br>
*[Wang Yifan](https://yifita.github.io/), Lukas Rahmann, [Olga Sorkine-Hornung](https://igl.ethz.ch/people/sorkine/).*<br>
ICLR 2022. [[Paper](https://arxiv.org/abs/2106.05187)] [[Project](https://yifita.github.io/publication/idf/)] [[Code](https://github.com/yifita/idf)]

* **Neural Volumes: Learning Dynamic Renderable Volumes from Images.**<br>
*Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, Yaser Sheikh.*<br>
TOG 2019. [[Paper](https://arxiv.org/abs/1906.07751)] [[Code](https://github.com/facebookresearch/neuralvolumes)]

* **DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation.**<br>
*eong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, Steven Lovegrove.*<br>
CVPR 2019. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html)] [[Code](https://github.com/facebookresearch/DeepSDF)] 

* **Occupancy Networks: Learning 3D Reconstruction in Function Space.**<br>
*Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, Andreas Geiger.*<br>
CVPR 2019. [[Paper](https://arxiv.org/abs/1812.03828)] [[Project](https://avg.is.mpg.de/publications/occupancy-networks)] [[Code](http://avg.is.tuebingen.mpg.de/publications/occupancy-networks)]

## Neural Inverse Rendering (Neural De-rendering)

[Inverse rendering](https://github.com/weihaox/awesome-neural-rendering/blob/master/docs/NEURAL-INVERSE-RENDERING.md) is to infer underlying intrinsic components of a scene from rendered 2D images. These properties include shape (surface, depth, normal), material (albedo, reflectivity, shininess), and lighting (direction, intensity), which can be further used to render photorealistic images. 

* **NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination.**<br>
*[Xiuming Zhang](http://people.csail.mit.edu/xiuming/), [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/), [Boyang Deng](https://boyangdeng.com/), [Paul Debevec](http://www.pauldebevec.com/), [William T. Freeman](http://billf.mit.edu/), [Jonathan T. Barron](https://jonbarron.info/).*<br>
SIGGRAPH Asia 2021. [[Paper](https://arxiv.org/abs/2106.01970)] [[Project](http://people.csail.mit.edu/xiuming/projects/nerfactor/)] [[Code](https://github.com/google/nerfactor)]

* **Extracting Triangular 3D Models, Materials, and Lighting From Images.**<br>
*[Jacob Munkberg](https://research.nvidia.com/person/jacob-munkberg), [Jon Hasselgren](https://research.nvidia.com/person/jon-hasselgren), [Tianchang Shen](http://www.cs.toronto.edu/~shenti11/), [Jun Gao](http://www.cs.toronto.edu/~jungao/), [Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Thomas Müller](https://research.nvidia.com/person/thomas-mueller), [Sanja Fidler](https://www.cs.toronto.edu/~fidler/).*<br>
CVPR 2022. [[Paper](http://arxiv.org/abs/2111.12503)] [[Code](https://github.com/NVlabs/nvdiffrec)] [[Project](https://nvlabs.github.io/nvdiffrec/)]

* **Modeling Indirect Illumination for Inverse Rendering.**<br>
*Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, Xiaowei Zhou.*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2204.06837)]

* **IRISformer: Dense Vision Transformers for Single-Image Inverse Rendering in Indoor Scenes.**<br>
*Rui Zhu, Zhengqin Li, Janarbek Matai, Fatih Porikli, Manmohan Chandraker.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_IRISformer_Dense_Vision_Transformers_for_Single-Image_Inverse_Rendering_in_Indoor_CVPR_2022_paper.html)]

* **De-rendering 3D Objects in the Wild.**<br>
*[Felix Wimbauer](https://www.linkedin.com/in/felixwimbauer/), [Shangzhe Wu](https://elliottwu.com/), [Christian Rupprecht](https://chrirupp.github.io/).*<br>
CVPR 2022. [[Paper](https://arxiv.org/abs/2201.02279)] [[Project](https://www.robots.ox.ac.uk/~vgg/research/derender3d/)] [[Code](https://github.com/Brummi/derender3d)]

* **GAN2X: Non-Lambertian Inverse Rendering of Image GANs.**<br>
*[Xingang Pan](https://xingangpan.github.io/), [Ayush Tewari](https://ayushtewari.com/), [Lingjie Liu](https://lingjie0206.github.io/), [Christian Theobalt](http://www.mpi-inf.mpg.de/~theobalt/).*<br>
3DV 2022. [[Paper](https://arxiv.org/abs/2206.09244)] [[Project](https://people.mpi-inf.mpg.de/~xpan/GAN2X/)]

* **PhySG: Inverse Rendering with Spherical Gaussians for Physics-based Material Editing and Relighting.**<br>
*Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, Noah Snavely.*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2104.00674)] [[Project](https://kai-46.github.io/PhySG-website/)]

* **Unified Shape and SVBRDF Recovery using Differentiable Monte Carlo Rendering.**<br>
*[Fujun Luan](https://www.cs.cornell.edu/~fujun/), [Shuang Zhao](https://www.shuangz.com/), [Kavita Bala](https://www.cs.cornell.edu/~kb/), [Zhao Dong](http://flycooler.com/).*<br>
EGSR 2021. [[Paper](https://www.cs.cornell.edu/~fujun/files/egsr2021/paper.Paper)] [[Project](https://luanfujun.github.io/InverseMeshSVBRDF/)] [[Video](https://youtu.be/u9HqKGqvJhQ?t=8404)]

* **Invertible Neural BRDF for Object Inverse Rendering.**<br>
*Zhe Chen, Shohei Nobuhara, Ko Nishino.*<br>
ECCV 2020. [[Paper](https://arxiv.org/abs/2008.04030)] [[Code](https://github.com/chenzhekl/iBRDF)]

* **Polarimetric Multi-View Inverse Rendering.**<br>
*Jinyu Zhao, Yusuke Monno, Masatoshi Okutomi.*<br>
ECCV 2020. [[Paper](https://arxiv.org/abs/2007.08830)]

* **Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image.**<br>
*[Zhengqin Li](http://sites.google.com/a/eng.ucsd.edu/zhengqinli/), [Mohammad Shafiei](https://www.linkedin.com/in/mohammadshafiei/), [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/), [Kalyan Sunkavalli](http://www.kalyans.org/), [Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/).*<br>
CVPR 2020.[[Paper](https://drive.google.com/file/d/18zG1kzVpL9XsEVBK95hbpnB-FMlChRXP/view?usp=sharing)] [[Project](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/)] [[Code](https://github.com/lzqsd/InverseRenderingOfIndoorScene)]

* **DRWR: A Differentiable Renderer without Rendering for Unsupervised 3D Structure Learning from Silhouette Images.**<br>
*Zhizhong Han, Chao Chen, Yu-Shen Liu, Matthias Zwicker.*<br>
ICML 2020. [[Paper](https://arxiv.org/abs/2007.06127)]

* **Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer.**<br>
*Wenzheng Chen, Jun Gao, Huan Ling, Edward J. Smith, Jaakko Lehtinen, Alec Jacobson, Sanja Fidler.*<br>
NeurIPS 2019. [[Paper](https://arxiv.org/abs/1908.01210)] [[Code](https://github.com/nv-tlabs/DIB-R)]

* **InverseRenderNet: Learning Single Image Inverse Rendering.**<br>
*Ye Yu, William A. P. Smith.*<br>
CVPR 2019. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_InverseRenderNet_Learning_Single_Image_Inverse_Rendering_CVPR_2019_paper.html)] [[Code](https://github.com/YeeU/InverseRenderNet)] [[IIW Dataset](http://opensurfaces.cs.cornell.edu/publications/intrinsic/#download)] 

## Neural Rerendering

* **Hybrid Neural Fusion for Full-frame Video Stabilization.**<br>
*[Yu-Lun Liu](https://www.cmlab.csie.ntu.edu.tw/~nothinglo/), [Wei-Sheng Lai](https://www.wslai.net/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/).*<br>
ICCV 2021. [[Paper](https://arxiv.org/abs/2102.06205)] [[Code](https://alex04072000.github.io/NeRViS/)]

* **Neural Lumigraph Rendering.**<br>
*Petr Kellnhofer, Lars Jebe, Andrew Jones, Ryan Spicer, Kari Pulli, Gordon Wetzstein.*<br>
CVPR 2021. [[Paper](https://arxiv.org/abs/2103.11571)] [[Project](http://www.computationalimaging.org/publications/nlr/)] [[Data](https://drive.google.com/file/d/1BBpIfrqwZNYmG1TiFljlCnwsmL2OUxNT/view?usp=sharing)]

* **Neural Re-Rendering of Humans from a Single Image.**<br>
*Kripasindhu Sarkar, Dushyant Mehta, Weipeng Xu, Vladislav Golyanik, Christian Theobalt.*<br>
ECCV 2020. [[Paper](https://arxiv.org/abs/2101.04104)]

* **Neural Rerendering in the Wild.**<br>
*Moustafa Meshry, Dan B Goldman, Sameh Khamis, Hugues Hoppe, Rohit Pandey, Noah Snavely, Ricardo Martin-Brualla.*<br>
CVPR 2019. [[Paper](https://arxiv.org/abs/1904.04290)]

* **Revealing Scenes by Inverting Structure from Motion Reconstructions.**<br>
*Francesco Pittaluga, Sanjeev J. Koppal, Sing Bing Kang, Sudipta N. Sinha.*<br>
CVPR 2019. [[Paper](https://arxiv.org/abs/1904.03303)]

Datasets
------------------

Summary of popular 3D-aware image synthesis datasets.

### Multi-view image collections

The images are rendered or collected according to different experimental settings, such as Synthetic-NeRF dataset, the DTU dataset, and the Tanks and Temples dataset for general purposes, the crowded Phototourism dataset for varying lighting conditions, the Blender Forward Facing (BLEFF) dataset to benchmark camera parameter estimation and novel view synthesis quality, and the San Francisco Alamo Square Dataset for large-scale scenes.

Examples of multi-view image datasets.

| **dataset**       | **published in**     |     **# scene**       |   **# samples per scene**   |   **range (m × m)**    |   **resolution**    |   **keyword**          |
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

Citation
------------------

If this repository benefits your research, please consider citing our paper.

```bibtex
  @inproceedings{xia2023survey,
    title={A Survey on Deep Generative 3D-aware Image Synthesis},
    author={Xia, Weihao and Xue, Jing-Hao},
    booktitle={ACM Computing Surveys (CSUR)},
    year={2023}
  }
```

License
------------------

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
