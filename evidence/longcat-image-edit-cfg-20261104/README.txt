LongCat-Image-Edit Cache-DiT: four-round paired SSIM evidence

This branch is an evidence attachment, not part of the upstream code diff.
All 24 pairs, including all three regressions, are included in paired-results.csv
and summary.json. The JPEG preview contains actual model-generated outputs;
"uncached reference" means generated output with caching disabled, not ground truth.
Display figures are not used to compute metrics. Metrics used the unmodified
RGB 1024x1024 PNGs; per-image hashes are preserved in summary.json and step-counts.csv.

Code / model identity
- Base: fc7ea43775b840ecc2b79addec9cd138065db5f0
- Submitted fix: b301911953f45ce607dcf66b1300b554301c89bd
- Model: meituan-longcat/LongCat-Image-Edit
- Model revision: 7b54ef423aa7854be7861600024be5c56ab7875a
- Frozen experiment patch SHA256:
  4b054d3e195cf6f989afebfe045c85b0af67795acbc2e42a5b7b8ea48e134d31
- Frozen plan SHA256:
  a606a7ae68c78e471183b0a9ab529b373852b50fb7ff355c73181edb4571f867

The submitted runtime files match the measured patch byte-for-byte. After the
image study, only the unit-test request fixture changed to the real
OmniDiffusionRequest type with seed=0. The related test suite was rerun:
95 passed, 12 deselected, 20 warnings in 13.16 s. See submitted-runtime.json
and unit-tests.log. The original experiment patch hash intentionally remains
unchanged; it is not presented as the checksum of the final submission patch.

Environment
One NVIDIA A800 80GB PCIe, driver 580.65.06, CUDA 13.0, cuDNN 9.25.1,
Python 3.12.3, vLLM 0.33.0, torch 2.14.0, diffusers 0.40.0,
transformers 5.19.0 and cache-dit 1.5.0. Full measured versions and component
dtypes are in environment.json. All three variants used the same environment,
checkpoint, input bytes, request order, seeds, preprocessing and decoding.

Experiment
3 input images x 2 edits x seeds [7, 42, 123, 2026] = 24 primary pairs.
50 steps, 1024x1024, guidance_scale=1.0, negative_prompt="", eager execution,
default cache_config={}. A separate fresh process ran each of:
  reference: original main, cache disabled
  baseline:  original main, Cache-DiT enabled
  fixed:     patched main, Cache-DiT enabled
Each seed round starts with an excluded 2-step sequential-CFG warmup.
Three anchor requests were repeated after each variant: all nine repeats are
pixel-identical and excluded from the 24-pair mean. There were 72 primary,
9 repeated and 12 warmup PNGs. Every primary request executed 50 transformer
forwards. Both cached contexts counted 25 steps before the fix and 50 after it.

Primary SSIM: scikit-image, RGB, channel_axis=2, data_range=255,
gaussian_weights=True, sigma=1.5, use_sample_covariance=False.
Each cached image is compared to its same-request uncached output.
Repository TorchMetrics SSIM is reported separately; all 24 change directions agree.

Results
seed     before       after        delta         improved / regressed
7        0.914620     0.930735     +0.016114       6 / 0
42       0.929888     0.941851     +0.011963       5 / 1
123      0.904172     0.923079     +0.018907       6 / 0
2026     0.922538     0.929585     +0.007047       4 / 2
all      0.917805     0.931313     +0.013508      21 / 3

Regressions:
brooch_green_s2026: 0.853323 -> 0.840615, delta -0.012708
brooch_table_s2026: 0.934920 -> 0.926944, delta -0.007976
brooch_table_s42:   0.918319 -> 0.916057, delta -0.002262

The seed-cluster bootstrap 95% interval is [+0.009314, +0.017511]. This is
exploratory: only four seed rounds, conditional on these six selected editing
tasks. It is not a guarantee for other inputs or devices. No prompt/seed/cache
tuning or runtime changes were made after inspecting the frozen study's scores.
No universal quality, speedup, or bleaching-fix claim is made.
Physical two-GPU CFG parallel and online serving were not exercised.

Reproduction assets (exact hashes are in plan.json)
- cat.png: model revision above, assets/test.png
- dog.png: base revision above,
  tests/assets/qwen_image_edit/qwen_image_edit_2511_test2.png
- brooch.png: base revision above,
  tests/assets/hunyuan_image3/hunyuan_image_instruct_ref.png
All prompts and parameters are in plan.json; the input images are unchanged
copies of those public model/repository assets.

Recorded reproduction scripts
The scripts preserve the paths used in the run. To reproduce them, use the
recorded directory layout below or adjust filesystem paths only:
  /root/autodl-tmp/omni-9826/runtime          matching Python environment
  /root/autodl-tmp/omni-9826/vllm-omni       fixed repository checkout
  /root/autodl-tmp/omni-9826/ssim-matrix     scripts, plan, inputs and outputs
  /root/autodl-fs/omni-9826/models/LongCat-Image-Edit   complete model snapshot

Populate ssim-matrix/inputs/{cat,dog,brooch}.png from the assets above. Extract
`git archive fc7ea43775b840ecc2b79addec9cd138065db5f0 vllm_omni pyproject.toml`
into ssim-matrix/baseline-src, then copy baseline-generated-version.py into
baseline-src/vllm_omni/_version.py. Keep audit_worker.py in the task directory
(or on PYTHONPATH). The fixed checkout's three runtime hashes must match plan.json.

Activate the matching runtime and the cuDNN 9.25.1 library used for the run;
the captured setup used OMP_NUM_THREADS=8, VLLM_WORKER_MULTIPROC_METHOD=spawn
and TOKENIZERS_PARALLELISM=false. run_all.sh reads ../env.sh for this setup.
Then, from the study directory, with a fresh outputs/ directory:

  bash run_all.sh
  PYTHONPATH=../vllm-omni python analyze_matrix.py
  python make_contact_sheets.py

For the analyzer's previous single-case cross-check, place the included
previous-single-case/{uncached,audited,fixed_audited}/results.json under
../results/{uncached,audited,fixed_audited}/results.json. These are prior-run
hash records, not additional samples in the 24-pair statistics.

Additional diagnostic
dispatch-summary.json contains an independent two-step, seed-7, 1024x1024
no-CFG cat-to-dog trace for each variant. Effective attention implementations,
actual attention operator/kernel names and counts, and output pixels agree.
This diagnostic is excluded from primary SSIM and timings; it is not a trace
of every workload shape. Main-run attention metadata returned an empty list
and is not used as proof of actual kernel selection.

Files not published in this compact attachment
All 93 original PNGs, three full CPU/CUDA traces and full generation logs were
retained in a local 183,319,446-byte evidence archive. Archive SHA256:
0c5d10993ed5420f10d5d8fbc5a742c0878d017bb9bdcc4509600dc533fdfc78
That archive is not linked here as a public download. This attachment contains
all per-case numerical data, hashes, the complete frozen test contract,
reproduction scripts, regression preview and test/lint logs.

AI assistance: OpenAI Codex assisted with the implementation, source review,
regression tests, A800 experiments, data analysis and PR preparation.
