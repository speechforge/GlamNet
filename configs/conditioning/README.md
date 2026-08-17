These files are overlay configs. Use them after the unchanged baseline config, for example:

```bash
python train.py \
  --config configs/baseline/vocos_baseline.yaml \
  --config configs/conditioning/full_boundary_aware.yaml
```

The manifest paths and vocabulary sizes marked `???` must match generated JSONL
manifests and `.npz` frame-label sidecars. These overlays keep the Fourier head,
iSTFT synthesis, discriminators, and non-causal ConvNeXt backbone family unchanged.
