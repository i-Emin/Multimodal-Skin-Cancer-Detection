# Manifest provenance and version limit

The reported Colab runs used the authors' `train-set-metadata.csv` and `valid-set-metadata.csv` from the upstream repository's `main` branch on 17 November 2026. The run artifacts record source SHA-256, package versions, configuration, and the observed manifest counts, but **they did not save the manifest bytes or their hashes at run time**. An exact retrospective data-version guarantee is therefore unavailable.

At review on 17 November 2026, the upstream `main` head was [`4e712bc676603f4bef6ba2431b95ff83cf9fbcf5`](https://github.com/mmu-dermatology-research/isic-dicm-17k/commit/4e712bc676603f4bef6ba2431b95ff83cf9fbcf5), authored 11 November 2025. Its manifest Git blob IDs are:

| File | Git blob SHA-1 | Pinned source |
|---|---|---|
| Train manifest | `9053a0e2dea04d218de743d6a4753dc3262e6868` | [CSV](https://raw.githubusercontent.com/mmu-dermatology-research/isic-dicm-17k/4e712bc676603f4bef6ba2431b95ff83cf9fbcf5/train-set-metadata.csv) |
| Validation manifest | `80e0190526935b839da342795f4470c623477e12` | [CSV](https://raw.githubusercontent.com/mmu-dermatology-research/isic-dicm-17k/4e712bc676603f4bef6ba2431b95ff83cf9fbcf5/valid-set-metadata.csv) |

These public upstream IDs provide a stable reference for future reruns. They should not be represented as hashes observed during the completed Colab runs. The official image bundle was downloaded from the ISIC-DICM-17K collection; the report's 17,060-image and metadata counts are the recorded audit, not a byte-level archive digest. No raw metadata, images, or per-image predictions are copied into this public repository.

