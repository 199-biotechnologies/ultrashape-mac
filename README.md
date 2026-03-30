<div align="center">

# UltraShape for Mac

**Run UltraShape 1.0 3D shape generation natively on Apple Silicon.**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/ultrashape-mac?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/ultrashape-mac/stargazers)
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1--M4-000000?style=for-the-badge&logo=apple&logoColor=white)](https://support.apple.com/en-us/116943)
[![License](https://img.shields.io/badge/License-Tencent_Hunyuan-blue?style=for-the-badge)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2512.21185-b31b1b?style=for-the-badge)](https://arxiv.org/pdf/2512.21185)

UltraShape 1.0 generates high-fidelity 3D meshes from images via geometric refinement. The original requires NVIDIA CUDA. This port runs on your Mac.

[Why This Exists](#why-this-exists) | [Install](#install) | [Quick Start](#quick-start) | [How It Works](#how-it-works) | [Features](#features) | [Performance](#performance) | [Contributing](#contributing) | [License](#license)

</div>

---

## Why This Exists

[UltraShape 1.0](https://github.com/PKU-YuanGroup/UltraShape-1.0) by PKU-Yuan-Lab is a state-of-the-art 3D shape generation model. It takes a coarse mesh and refines it into a high-fidelity 3D shape with sharp geometric detail. The problem: it only runs on NVIDIA CUDA GPUs.

This port replaces every CUDA-only dependency with cross-platform alternatives so the entire pipeline runs on Apple Silicon via Metal Performance Shaders (MPS). No NVIDIA hardware required. CUDA support is preserved for users who have it.

| Component | Original (CUDA-only) | This Port |
|---|---|---|
| Marching Cubes | `cubvh` | `scikit-image` fallback |
| Attention | `flash_attn` | PyTorch `scaled_dot_product_attention` |
| SageAttention | Required | Optional, graceful fallback |
| torch_cluster | CUDA wheels only | Pure PyTorch FPS fallback |
| Device detection | CUDA hardcoded | Auto-detect CUDA / MPS / CPU |
| Autocast dtype | `bfloat16` | `float16` for MPS compatibility |

## Install

### Apple Silicon Mac (M1/M2/M3/M4)

```bash
git clone https://github.com/199-biotechnologies/ultrashape-mac.git
cd ultrashape-mac

python3 -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install dependencies (torch must be installed first)
pip install -r requirements.txt
# Note: 'diso' will fail to install (CUDA-only) - this is expected
```

### NVIDIA GPU (Linux/Windows)

```bash
git clone https://github.com/199-biotechnologies/ultrashape-mac.git
cd ultrashape-mac

conda create -n ultrashape python=3.10
conda activate ultrashape

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional CUDA accelerators
pip install flash_attn==2.8.3
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
```

### Download Model Weights

```bash
huggingface-cli download infinith/UltraShape --local-dir ~/.cache/hy3dgen/infinith/UltraShape
```

## Quick Start

UltraShape refines a coarse mesh into a detailed 3D shape. You need an input image and a coarse mesh (generate one with [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)).

**On Mac (MPS):**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PWD python scripts/infer_dit_refine.py \
    --image path/to/image.png \
    --mesh path/to/coarse_mesh.glb \
    --ckpt ~/.cache/hy3dgen/infinith/UltraShape/ultrashape_v1.pt \
    --output_dir outputs \
    --num_latents 4096 \
    --chunk_size 1024 \
    --octree_res 128 \
    --steps 12 \
    --low_vram
```

**On NVIDIA GPU:**

```bash
PYTHONPATH=$PWD python scripts/infer_dit_refine.py \
    --image path/to/image.png \
    --mesh path/to/coarse_mesh.glb \
    --ckpt ~/.cache/hy3dgen/infinith/UltraShape/ultrashape_v1.pt \
    --output_dir outputs
```

**Gradio Web UI:**

```bash
python scripts/gradio_app.py --ckpt path/to/checkpoint.pt
```

## How It Works

UltraShape 1.0 uses a DiT (Diffusion Transformer) architecture to refine coarse 3D meshes. The pipeline:

1. **Input** -- an image and a coarse mesh (from Hunyuan3D-2.1 or similar)
2. **Encode** -- the VAE encodes the mesh into a latent representation
3. **Condition** -- the image conditioner extracts visual features
4. **Refine** -- the DiT denoises the latent, guided by image features
5. **Decode** -- the VAE decodes refined latents back into a high-fidelity mesh

The Mac port replaces CUDA kernels with platform-agnostic PyTorch ops. MPS acceleration handles the heavy lifting on Apple Silicon. CPU offloading (`--low_vram`) keeps memory usage manageable on machines with less unified memory.

## Features

- **Native Apple Silicon support** via MPS (Metal Performance Shaders)
- **Auto device detection** -- CUDA, MPS, or CPU, picked automatically
- **Low VRAM mode** -- CPU offloading for memory-constrained machines
- **Gradio web UI** for interactive mesh refinement
- **Full CUDA support preserved** for NVIDIA users
- **Configurable inference** -- tune steps, latent count, chunk size, and resolution

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--steps` | 50 | Inference steps (12 for faster generation) |
| `--num_latents` | 32768 | Latent tokens (reduce to 4096-8192 if OOM) |
| `--chunk_size` | 8000 | Processing chunk size (reduce to 1024-2048 if OOM) |
| `--octree_res` | 1024 | Marching cubes resolution |
| `--low_vram` | False | Enable CPU offloading |

### Mac Tips

- Always set `PYTORCH_ENABLE_MPS_FALLBACK=1` -- some PyTorch ops lack MPS implementations
- Apple Silicon shares RAM between CPU and GPU. Close other apps to free memory.
- First inference is slower due to MPS kernel compilation
- Add the project root to PYTHONPATH: `PYTHONPATH=$PWD python ...`

## Performance

| Platform | Time per Mesh |
|---|---|
| NVIDIA RTX 4090 | ~2-3 min (with flash_attn) |
| Apple M3 Max (64GB) | ~10-15 min |
| Apple M1 (16GB) | ~30+ min (reduced settings) |
| CPU only | 1+ hour |

Tested Mac settings (88GB unified memory): `--num_latents 4096 --chunk_size 1024 --octree_res 128 --steps 12 --low_vram`

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project uses the [Tencent Hunyuan Community License](LICENSE). See the license file for full terms.

## Citation

```bibtex
@article{jia2025ultrashape,
    title={UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement},
    author={Jia, Tanghui and Yan, Dongyu and Hao, Dehao and Li, Yang and Zhang, Kaiyi and He, Xianyi and Li, Lanjiong and Chen, Jinnan and Jiang, Lutao and Yin, Qishen and Quan, Long and Chen, Ying-Cong and Yuan, Li},
    journal={arxiv preprint arXiv:2512.21185},
    year={2025}
}
```

## Acknowledgements

- [UltraShape 1.0](https://github.com/PKU-YuanGroup/UltraShape-1.0) by PKU-Yuan-Lab
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) by Tencent
- [LATTICE](https://arxiv.org/abs/2512.03052) for inspiring the core methodology

---

<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [199 Biotechnologies](https://github.com/199-biotechnologies) | [Paperfoot AI](https://paperfoot.ai)

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/ultrashape-mac?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/ultrashape-mac/stargazers)
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
