# Data directory

Download ISIC-DICM-17K from the [official ISIC Archive collection](https://api.isic-archive.com/doi/isic-dicm-17k/). The complete bundle is approximately 6.9 GB compressed and contains 17,060 dermoscopic images with metadata. The starter Colab notebook downloads it into `/content/isic-dicm-17k` and audits the actual file layout and CSV columns before any model training.

Do not commit images, raw metadata, extracted archives, patient information, or large checkpoints here. The collection lists CC-0, CC-BY, and CC-BY-NC licenses across its content; check item attribution and restrictions before reuse.

This tracked README keeps the `data/` directory visible in GitHub. All other files placed here are ignored by the repository's `.gitignore`.

