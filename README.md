# 2024 Update
We provide Google drive link for our dataset download.

# multi-view-3DGPR
The referencen code of paper **Multi-View Fusion and Distillation for Subgrade Distresses Detection based on 3D-GPR**
![image](https://github.com/zhouchunpong/multi-view-3DGPR/assets/6890539/71a8494a-c189-45e1-a458-681ea5661a30)
If you want to known more about this dataset and new method, please read our paper [link](https://arxiv.org/abs/2308.04779).

## Abstract
The 3D ground-penetrating radar (3D-GPR) has been widely used for the subgrade distress detection. To enhance the efficiency and accuracy of detection, pioneering studies have attempted to adopt automatic detection techniques such as deep learning. However, existing works typically rely on traditional 1D A-scan, 2D B-scan or 3D C-scan data of the GPR, resulting in either insufficient spatial information or high computational complexity.  To address these challenges,  we introduce a novel methodology for the subgrade distress detection task by leveraging the multi-view information from the vanilla 3D-GPR data. Consequently, we construct a real multi-view image dataset derived from the vanilla 3D-GPR data for the detection task,  providing richer spatial information compared to A-scan and B-scan data, and reducing data computational complexity compared to C-scan data. Subsequently, we develop  a novel Multi-View Fusion and Distillation framework, called GPR-MVFD,  specifically designed to optimally utilize the multi-view GPR dataset. This framework ingeniously incorporates the attention-
based fusion and multi-view distillation to facilitate significant feature extraction for subgrade distresses. In addition, a self-adaptive learning mechanism is adopted to stabilize the model training and prevent performance degeneration in each branch. Extensive experiments demonstrate the value of newly constructed GPR dataset and showcase the effectiveness and efficiency of our proposed GPR-MVFD. Notably, our framework outperforms not only the existing GPR baselines, but also the state-of-the-art methods in the fields of multi-view learning, multi-modal learning, and knowledge distillation. 



# multi-view-3DGPR Dataset



<img src="https://github.com/zhouchunpong/multi-view-3DGPR/assets/6890539/efecdad8-08b3-48f1-b845-077b9f7c08c9"  width="50%" />


The dataset is available at now: 

Baidu Drive Link：https://pan.baidu.com/s/14uZ6F0NbQxgwfaWTERQX2Q 
Passwd：2023

Google Drive Link：https://drive.google.com/drive/folders/1TbZCAUq7GEWRk7dUo3CmuhyE1DUnKoen?usp=sharing



# Requirements
* python 3.7
* pytorch 1.11.0
* cuda 11.3


## Usages

To train the model described in the paper, run the following command:

```
python ./code/train_DenseNet121_MVFD.py
```

To evaluate the trained model, run the following command:

```
python ./code/test_DenseNet121_MVFD.py
```




## Citation
If you find our paper or dataset useful, please give us a citation.
```bash
@article{zhou2023multi,
  title={Multi-View Fusion and Distillation for Subgrade Distresses Detection based on 3D-GPR},
  author={Zhou, Chunpeng and Ning, Kangjie and Wang, Haishuai and Yu, Zhi and Zhou, Sheng and Bu, Jiajun},
  journal={arXiv preprint arXiv:2308.04779},
  year={2023}
}
```
