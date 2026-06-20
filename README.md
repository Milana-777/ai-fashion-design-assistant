# AI Fashion Design Assistant
8-week Generative AI internship project at Aarivya Labs.
Week 1: Dataset pipeline | Week 2: Stable Diffusion generation

### Week 3 — Style Control with ControlNet
- Integrated Canny edge ControlNet with Stable Diffusion v1.5
- Built sketch-to-design pipeline (sketch → edge map → generated garment)
- Added OpenPose-based pose control for fashion model poses
- Compared controlled vs uncontrolled generation using CLIP scores
- Result: average CLIP score improved from 23.1 to 26.1 (+3.0) with ControlNet

### Week 4 — LoRA Fine-Tuning for Brand Style
- Curated 3 brand-specific datasets: minimalist, streetwear, luxury
- Trained lightweight LoRA adapters (rank=2) optimized for CPU training
- Implemented prompt-token style switching (e.g. <minimalist-style>)
- Evaluated brand consistency using CLIP similarity scoring
- Minimalist adapter achieved final training loss of 0.0835
