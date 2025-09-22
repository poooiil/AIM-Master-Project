# 🎼 Beat-Aware Diffusion for Music-Driven Choral Conducting Motion Generation

![teaser](docs/teaser.png)  
*Music-driven motion generation has seen growing interest, while conducting remains less explored. We study this task in the context of choral repertoire. Our method uses a phase-based beat cue that locates each frame within the current beat and a diffusion model conditioned on musical features to promote timing consistency and natural upper-body motion. Evaluations on held-out pieces indicate clearer beat alignment and plausible gestures compared with representative baselines.*  

---

## 📂 Dataset & Pretrained Models

### 1. Maestro3D Dataset  
Download all `.npy` files and place them into the `demo/` folder:  
🔗 [Google Drive Link](https://drive.google.com/drive/folders/1x-oST6VXu-AKbwYFuMaPwcB28GCe0rOg?usp=sharing)


## Dataset
使用时把里面的所有npy放到demo文件里
https://drive.google.com/drive/folders/1x-oST6VXu-AKbwYFuMaPwcB28GCe0rOg?usp=sharing

pth文件，放到weight文件夹里
https://drive.google.com/file/d/1vTD9s6JJV9mT7WieeuiGyGrNK6AX0d4l/view?usp=sharing

body_models/smplh/SMPLH_MALE.pkl
https://drive.google.com/file/d/1hMDRkFnSqTQTokeKdojEA38FbsvaQEQi/view?usp=sharing

body_models/smpl/SMPL_NEUTRAL.pkl
https://drive.google.com/file/d/1jp3ZquJxN9944JwAq9a7c00jbE12WKH0/view?usp=sharing

