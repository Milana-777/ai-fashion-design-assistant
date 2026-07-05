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

### Week 5 — Intelligent Style Recommendation (RAG)
- Built fashion knowledge base with 8 trends, 6 style personas, 6 color palettes
- Embedded 200 fashion images using CLIP ViT-B-32 into 512-dimensional vectors
- Stored and queried embeddings using ChromaDB vector database
- Built RAG pipeline: text query → CLIP embedding → ChromaDB search → trend matching
- Processed 5 style queries with relevant image and trend recommendations

### Week 6 — Gradio Creative Studio
- Built 4-tab interactive Gradio creative studio
- Sketch2Design: upload hand-drawn sketch → AI generates fashion photo
- StyleMixer: select brand style + garment + color → styled design output
- WardrobeGen: choose occasion + season + persona → complete outfit generation
- Trend explorer: type style query → RAG retrieves trends + similar images
- Live demo runs locally at http://127.0.0.1:7860
