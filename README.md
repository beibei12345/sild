<h1 align="center">Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts (SILD)</h1>
<p align="center">
    <a href="https://github.com/wondergo2017/sild"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/154b90fcc9ba3dee96779c05c3108908-Abstract-Conference.html"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=blue"> </a>
    <a href="./paper/NeurIPS23_SILD_poster.pdf"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
</p>

This repository contains the code implementation of SILD as described in the paper: [Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts](https://proceedings.neurips.cc/paper_files/paper/2023/hash/154b90fcc9ba3dee96779c05c3108908-Abstract-Conference.html) (NeurIPS 2023).

## Introduction

<p align="center"><img src="./paper/framework.png" width=85% height=85%></p>
<p align="center"><em>Figure 1.</em> The framework of SILD.</p>

Dynamic graph neural networks (DyGNNs) currently struggle with handling distribution shifts that are inherent in dynamic graphs.
Existing work on DyGNNs with out-of-distribution settings only focuses on the time domain, failing to handle cases involving distribution shifts in the spectral domain.

In this paper, we discover that there exist cases with distribution shifts unobservable in the time domain while observable in the spectral domain, 
and propose to study distribution shifts on dynamic graphs in the spectral domain for the first time.

To this end, we propose Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts (SILD), which can handle distribution shifts on dynamic graphs by capturing and utilizing invariant and variant spectral patterns. Specifically, we first design a DyGNN with Fourier transform to obtain the ego-graph trajectory spectrums, allowing the mixed dynamic graph patterns to be transformed into separate frequency components. We then develop a disentangled spectrum mask to filter graph dynamics from various frequency components and discover the invariant and variant spectral patterns. Finally, we propose invariant spectral filtering, which encourages the model to rely on invariant patterns for generalization under distribution shifts. Experimental results on synthetic and real-world dynamic graph datasets demonstrate the superiority of our method for both node classification and link prediction tasks under distribution shifts. 

The framework is shown in Figure 1.



## Installation
We have tested our codes with the following requirements:  
- Python == 3.9
- Pytorch == 1.12.1+cu113
- Pytorch-Geometric == 2.0.4+cu113

or 

- Python == 3.9
- Pytorch == 2.0.1+cu117
- Pytorch-Geometric == 2.3.0+cu117

Take the latter as an example. Please follow the following steps to create a virtual environment and install the required packages.

Clone the repository:
```
git clone git@github.com:wondergo2017/sild.git
cd sild
```

Create a virtual environment:
```
conda create --name sild python=3.9 -y
conda activate sild
```

Install dependencies:

```
pip install torch==2.0.1
pip install torch_geometric==2.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Install DIDA as a library:
```
pip install -e DIDA/
```

Install this repo as a library:
```
pip install -e .
```

The datasets can be downloaded from [google drive](https://drive.google.com/file/d/1GMuWLJUg5n-lYO7gH0RXn7Rr3b4WG18Z/view?usp=sharing).
## File Organization & Datasets

The files in this repo are organized as follows:

```
\data                       # datasets introduced in sild
\DIDA                       # the original codes of the baseline DIDA
\sild
    \data_util              # data utils including data loading, processing, etc. 
    \misc                   # other utils                   
\paper                      # poster, framework, etc.                       
\scripts\
    \aminer                 # scripts for running exps on aminer datasets
        config.py          
        main.py             # main script
        model.py            
        utils.py
        parallel-rep.py     # the script to reproduce results
    \dida_data              # scripts for running exps on DIDA datasets 
        ...
    \syn_SBM                # scripts for running exps on synthetic node classification datasets.
        ...    
```

## Usage
An example to run the model is given as follows:
(you may change the gpus in parallel-rep.py)
```
cd scripts/aminer
python parallel-rep.py -t run
```
To show the results 
```
python parallel-rep.py -t show
```
Or if you want to run the model with your defined arguments
```
python main.py --X XXX
```


## Acknowledgements

The datasets and baseline scripts are modified from the publicly available repos or other resources. We sincerely appreciate their contributions to the research community.

## Reference

If you find our repo or paper useful, please star the repo or cite our paper:
```bibtex
@inproceedings{zhang2023spectral,
  title={Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts},
  author={Zhang, Zeyang and Wang, Xin and Zhang, Ziwei and Qin, Zhou and Wen, Weigao and Xue, Hui and Li, Haoyang and Zhu, Wenwu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Contact

If you have any questions, please feel free to contact us (zy-zhang20@mails.tsinghua.edu.cn) or (wondergo2017@gmail.com) 
