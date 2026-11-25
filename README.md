# FreqSSM-Lite — a runnable, CPU-only low-light enhancer in the spirit of ExpoMamba

A small reimplementation of the ExpoMamba ideas (U-Net + frequency-domain block + selective
state-space scan) that **needs only NumPy and Pillow**: no PyTorch, no CUDA, no `mamba_ssm`.
It trains on LOL-v2 Synthetic on a laptop CPU.

## Model (`lfs/model.py`, ~280K params at `--channels 16`)
| Part | What it does |
|---|---|
| U-Net, 3 scales | conv stem → 2 stride-2 downsamplings → bottleneck → nearest upsampling + skip concat |
| `ResBlock` | conv3×3 → LeakyReLU → conv3×3, residual |
| `FreqBlock` | adaptive Fourier filter: log-amplitude spectrum → 1×1 conv → gain `m = 1 + 0.5·tanh(·)`, output `Re(IFFT(FFT(x)·m))`; identity at init |
| `SSMBlock` | Mamba-style selective scan `h_t = a_t·h_{t-1} + (1-a_t)·u_t` with input-dependent decay `a = σ(W_a x)`, 4 scan directions (VMamba cross-scan), gate `silu(W_z x)` |
| output | `input + correction` (global residual) |

Loss: Charbonnier + 0.05 × L1 between FFT amplitude spectra. Optimiser: Adam, warmup + cosine LR,
gradient clipping 1.0. `lfs/autograd.py` is a small reverse-mode autograd; every op is checked
against finite differences in `tests/test_gradcheck.py`.

Differences from ExpoMamba (by design, to run on CPU): far fewer parameters, scan state size 1
per channel (diagonal), no HDR/exposure branches, no VGG/LPIPS loss, no dynamic patch schedule.

## Run
```bash
pip install -r requirements.txt            # numpy, pillow
python tests/test_gradcheck.py             # ~10 s, should print ALL OK
python train.py --data_root ../expomamba-llie/datasets/LOLv2_syn --name lolv2syn_c16
python test.py  --ckpt runs/lolv2syn_c16/best.npz --data_root ../expomamba-llie/datasets/LOLv2_syn --save_images
```
* `--data_root` must contain `Train/{Input,GT}` and `Test/{Input,GT}` (the HF `okhater/lolv2-synthetic` layout;
  test inputs `r<id>.png` are matched to GT `<id>.png` automatically).
* Data split: 850 train / 50 val (held out from Train, seed 3407) for model selection; the 100 Test pairs are used only by `test.py`.
* The first run caches the images as `.npy` in `~/.cache/lowlight-freqssm` (~0.8 GB).
* Training resumes automatically: re-run the same command. `--max_minutes N` stops cleanly after N minutes.
* Outputs in `runs/<name>/`: `log.csv`, `config.json`, `best.npz`, `last.npz`, `best.json`, and after testing
  `test_results.json`, `test_metrics.csv` (per image), `test_images/`.

Useful knobs: `--iters` (default 4000), `--patch 64`, `--batch 8`, `--channels 16`, `--lr 1e-3`.
