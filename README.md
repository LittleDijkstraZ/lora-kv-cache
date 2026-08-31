# LoRA-KV-Cache

Code for [Rethinking LoRA Memory Through the Lens of KV Cache Compression](https://arxiv.org/abs/2606.05698).

We study document-specific LoRA adapters and compressed KV caches as two
complementary forms of memory for document question answering. Experiments use
NarrativeQA and LongHealth with Qwen3-4B and Llama-3.1-8B-Instruct.

## Installation

The paper environment uses Python 3.10 and CUDA 12.8.

```bash
conda create -n lora-kv-cache python=3.10 -y
conda activate lora-kv-cache
bash scripts/install.sh
```

## Data

Download and prepare the evaluation subsets:

```bash
python scripts/prepare_data.py --dataset all
```

The script downloads and prepares
[NarrativeQA](https://github.com/deepmind/narrativeqa) and
[LongHealth](https://github.com/kbressem/LongHealth). Model checkpoints and
benchmark data are not stored in this repository.

## Experiments

Preview the commands without starting GPU jobs:

```bash
python scripts/run_experiments.py --group all
```

Run the experiments:

```bash
python scripts/run_experiments.py --group all --run
```

Available groups are `main`, `training-formats`, `compression-methods`, and
`target-modules`. Focused ablation groups include the required main baseline.
Use `--models` or `--datasets` to run a subset.

## Citation

```bibtex
@article{zuo2026rethinking,
  title   = {Rethinking LoRA Memory Through the Lens of KV Cache Compression},
  author  = {Zuo, Chunsheng and Wang, Liaoyaqi and Jurayj, William and Fleshman, William and Van Durme, Benjamin},
  journal = {arXiv preprint arXiv:2606.05698},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.05698}
}
```

## License

Released under the [Apache License 2.0](LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party attributions.
