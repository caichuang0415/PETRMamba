# Environment Setup
```bash
# 1.create a conda environment
conda create -n PETRmamba python=3.8 -y
conda activate PETRmamba
# 2.Install the Pytorch according to your device's cuda version: https://pytorch.org/get-started/previous-versions/
# for example：
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# 3.Install flash-attn (optional)
pip install flash-attn==0.2.2
# 4.Install causal-conv1d (optional)
pip install causal-conv1d>=1.2.0
# 5.Clone PETRMamba
git clone https://github.com/caichuang0415/PETRMamba.git
# 6.Install mmdet3d
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
cd ./PETRMamba
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
```