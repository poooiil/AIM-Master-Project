# 🎼 Beat-Aware Diffusion for Music-Driven Choral Conducting Motion Generation

![teaser](fig/model.png)  
*Music-driven motion generation has seen growing interest, while conducting remains less explored. We study this task in the context of choral repertoire. Our method uses a phase-based beat cue that locates each frame within the current beat and a diffusion model conditioned on musical features to promote timing consistency and natural upper-body motion. Evaluations on held-out pieces indicate clearer beat alignment and plausible gestures compared with representative baselines.*  

---

## 📂 Dataset & Pretrained Models

### 1. Our Dataset  
Download all `.npy` files and place them into the `demo/` folder:  
🔗 [Google Drive Link](https://drive.google.com/drive/folders/1x-oST6VXu-AKbwYFuMaPwcB28GCe0rOg?usp=sharing)

---

### 2. Pretrained Weights  
Download the pretrained model (`.pth`) and put it in the `weight/` folder:  
🔗 [Google Drive Link](https://drive.google.com/file/d/1vTD9s6JJV9mT7WieeuiGyGrNK6AX0d4l/view?usp=sharing)

---

### 3. Body Models  

- **SMPL-H (male)** → place in `body_models/smplh/`  
  🔗 [Download Link](https://drive.google.com/file/d/1hMDRkFnSqTQTokeKdojEA38FbsvaQEQi/view?usp=sharing)

- **SMPL (neutral)** → place in `body_models/smpl/`  
  🔗 [Download Link](https://drive.google.com/file/d/1jp3ZquJxN9944JwAq9a7c00jbE12WKH0/view?usp=sharing)



