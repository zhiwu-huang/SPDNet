# SPDNet-master
Zhiwu Huang and Luc Van Gool. A Riemannian Network for SPD Matrix Learning, In Proc. AAAI 2017. 

Version 1.0,  Copyright(c) November, 2017. 

Note that the copyright of the manopt toolbox is reserved by https://www.manopt.org/  

## Usage:

Step1: Place the used <a href="https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/AFEW_SPD_data.zip"> AFEW SPD data </a> under the folder "./data/afew/". (Note that the used <a href="https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/HDM05_SPDData.zip"> HDM05 </a> and <a href="https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/PaSC_SPDData.zip"> PaSC </a> SPD data are also publicly available.)

Step2: Launch spdnet_afew.m for a simple example.

## Related Work/Implementation:

1. Thanks to Oleg Smirnov who is Sr. Applied Scientist at Amazon, a <a href="https://github.com/master/tensorflow-manopt"> Tensorflow ManOpt </a> library is released to reproduce our SPDNet.

2. A NeurIPS 2019 paper "Riemannian batch normalization for SPD neural networks" develops batch normalization layer upon our SPDNet, with the official PyTorch code being publicly available at the 'Supplemental' tab of https://proceedings.neurips.cc/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html

3. A report "Second-order networks in PyTorch" is released at https://core.ac.uk/download/pdf/231946513.pdf

4. Thanks to Alireza Davoudi, there is another Python implementation for SPDNet at https://github.com/adavoudi/spdnet

5. A direct extension of our SPDNet for facial emotion recognition is published by CVPR workshop 2018, with the code being available at https://github.com/d-acharya/CovPoolFER 


## How to Cite <a name="How-to-Cite"></a>
If you find this project helpful, please consider citing us as follows:
```bash
@inproceedings{huang2017spdnet,
      title = {A Riemannian Network for SPD Matrix Learning},
      author    = {Huang, Zhiwu and
                   Van Gool, Luc},
      year = {2017},
      booktitle = {Association for the Advancement of Artificial Intelligence (AAAI)}
}


