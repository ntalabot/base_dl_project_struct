# Installation
Start by downloading the repository:
```bash
git clone https://github.com/ntalabot/base_dl_project_struct.git
cd base_dl_project_struct
```

## Conda environment
Create a conda environment for the project with your python version, then activate it.
```bash
conda create --name dl_project python=3.6
conda activate dl_project
``` 

Install the dependencies and pytorch with the correct cuda version:
```bash
conda install --file requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Replace `cudatoolkit=10.2` by `cpuonly` if you don't have a GPU.

Then install this package in editable mode through pip (by default as `src`, so best used within an environment):
```
#conda install pip
pip install -e .
```