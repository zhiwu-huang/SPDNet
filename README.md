# SPDNet-master
Zhiwu Huang and Luc Van Gool. A Riemannian Network for SPD Matrix Learning, In Proc. AAAI 2017. 

Version 1.0,  Copyright(c) November, 2017. 

Note that the copyright of the manopt toolbox is reserved by https://www.manopt.org/  

## Usage:

Step1: Place the AFEW SPD data under the folder "./data/afew/". The AFEW SPD data can be downloaded from 
1. https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/AFEW_SPD_data.zip
2. https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/HDM05_SPDData.zip
3. https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/PaSC_SPDData.zip


Step2: Launch spdnet_afew.m for a simple example.

If you find any bugs, please contact me via zhiwu.huang@vision.ee.ethz.ch

## Related work/implementation:

A NeurIPS 2019 paper "Riemannian batch normalization for SPD neural networks" develops batch normalization layer upon our SPDNet, with its official Pytorch code publicly available at the 'Supplemental' tab of https://proceedings.neurips.cc/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html

Thanks to Alireza Davoudi, there is another Python implementation for SPDNet at https://github.com/adavoudi/spdnet. 


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


