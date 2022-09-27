# SmartPath: an open-source automatic multimodal whole slidehistopathology imaging system

<div align="center">
  <img src="docs/github.png" width="700px" />
</div>

SmartPath is an open-source multimodal whole slide histopathology imaging system with the capabilities of automaticmodality switching, region of interest detection, and run-time image processing, empowered by deep learning. The acquisition program is written in [Jupyter Notebook/JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) and uses [Pycro-Manager](https://pycro-manager.readthedocs.io/en/latest/) for hardware control.  

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html).  
Open terminal or Anaconda Prompt in Windows.  
Install required packages:  
```
  $ conda env create --name smartpath --file env.yml
  $ conda activate smartpath
  $ pip install pycromanager
```  
Install [PyTorch](https://pytorch.org/get-started/locally/), pick GPU or CPU version.  

## Software requirement
Micro-manager 2.0 gamma, OpenScan, QuPath and FIJI. See [document](https://docs.google.com/document/d/1mAJuh3Eu8Bkt_IAWVCzh7XT79KacVCjf/edit?usp=sharing&ouid=111512958445591507194&rtpof=true&sd=true) for details.  

## How to use  
See step-by-step instructions in the [notebook](main.ipynb).

## Related repositories
Supervised and weakly supervised classifier for histological datasets: https://github.com/uw-loci/histo-classifier.  
Run-time image enhancement for laser scanning microscopy with self-supervised denoising and single-image super-resolution: https://github.com/uw-loci/lsm-run-time-enhancement.
