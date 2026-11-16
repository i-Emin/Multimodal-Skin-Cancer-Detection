"""Generate a Colab runner that calls the versioned training module."""
import json
import hashlib
from pathlib import Path


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": value.splitlines(keepends=True)}


baseline_source = Path('src/baseline.py').read_text(encoding='utf-8')
baseline_sha256 = hashlib.sha256(baseline_source.encode('utf-8')).hexdigest()
embedded_source = baseline_source.split('if __name__ == "__main__":')[0]

cells = [
    markdown("""# YZM5110 — Three baseline models

Run top to bottom on Colab. This notebook uses the [authors' published train and validation manifests](https://github.com/mmu-dermatology-research/isic-dicm-17k) and their `class` labels. The archive CSV's `diagnosis_1` is **not** used as a label or clinical feature.

**Interpretation limit:** The published train/validation manifests share 433 nonmissing patient IDs and 488 lesion IDs. Results here are **paper-split, reduced-backbone baselines**, not patient-independent clinical estimates or an exact reproduction of Ahammed et al.'s published score. The separate ISIC 2020 Kaggle test is not evaluated. See [data audit](https://github.com/i-Emin/Multimodal-Skin-Cancer-Detection/blob/main/reports/data_audit.md).

The fixed final epoch is reported. Outputs, including a SHA-256 digest of the embedded training source, settings, hardware, learning history, metrics, and predictions, are saved to your Google Drive. They are not automatically published to GitHub. The three models use the same seed, split, epoch count, batch size, and learning rate. Image-only and fusion use ImageNet-pretrained ResNet18; this smaller backbone and augmentation differ from the selected paper.
"""),
    code("""from pathlib import Path
import torch

print('Torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
if not torch.cuda.is_available():
    raise RuntimeError('Select a GPU runtime before running image models.')
"""),
    markdown("""## Dataset

If you already ran the audit notebook in the **same runtime**, this cell reuses `/content/isic-dicm-17k`. Otherwise it downloads and extracts the official 6.9 GB bundle. Colab's `/content` is temporary. Check the package terms on the [ISIC collection page](https://api.isic-archive.com/doi/isic-dicm-17k/).
"""),
    code("""import urllib.request, zipfile, shutil

data_dir = Path('/content/isic-dicm-17k')
image_dir = data_dir / 'images'
if not image_dir.is_dir() or not (data_dir / 'metadata.csv').is_file():
    archive = Path('/content/isic-dicm-17k.zip')
    if not archive.exists():
        print('Downloading official bundle (~6.9 GB)...', flush=True)
        urllib.request.urlretrieve(
            'https://isic-archive.s3.amazonaws.com/dois/10.34970-233480/isic-dicm-17k.zip', archive)
    print('Free disk GB:', round(shutil.disk_usage('/content').free / 1e9, 1))
    with zipfile.ZipFile(archive) as z:
        bad = [name for name in z.namelist() if Path(name).is_absolute() or '..' in Path(name).parts]
        if bad:
            raise ValueError(f'Unsafe archive paths: {bad[:3]}')
        z.extractall(data_dir)

images = list(image_dir.glob('*.jpg'))
print('JPG images:', len(images))
if len(images) != 17060:
    raise RuntimeError('Expected 17,060 JPG images. Check extraction before training.')
"""),
    markdown("""## Save results to Drive

Mount your own Drive. This saves result files and final model checkpoints under `MyDrive/YZM5110/results/baselines/`. The raw dataset is not copied there. Checkpoints stay in your Drive and must not be committed to the public repository. If Google asks for Drive access, review the prompt before accepting.
"""),
    code("""from google.colab import drive
drive.mount('/content/drive')
out_dir = Path('/content/drive/MyDrive/YZM5110/results/baselines')
out_dir.mkdir(parents=True, exist_ok=True)
print('Output:', out_dir)
"""),
    markdown("""## Embedded versioned training code

The cell below is generated from `src/baseline.py`; do not edit it for this run. Its SHA-256 digest is saved in each `run.json` so results can be traced to exact code. The script version in GitHub is intended to match this cell once the repository sync completes.
"""),
    code(embedded_source + f"\nSOURCE_SHA256 = '{baseline_sha256}'\nprint('Training source SHA-256:', SOURCE_SHA256)\n"),
    markdown("""## Run the three matched baselines

Metadata-only runs first as a quick pipeline check. Image-only and fusion take longer. The code prints one row per epoch. Stop and share the full error if any run fails. Do not report a numerical result until all output files are written.
"""),
    code("""import argparse

EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEED = 42

for mode in ('metadata', 'image', 'fusion'):
    args = argparse.Namespace(mode=mode, data_dir=str(data_dir), output_dir=str(out_dir),
                              epochs=EPOCHS, batch_size=BATCH_SIZE,
                              lr=LEARNING_RATE, seed=SEED, workers=2,
                              save_checkpoint=True)
    print('RUNNING:', mode, flush=True)
    run(args)
"""),
    markdown("""## Inspect measured results

These are your measured validation values. The notebook does not contain fabricated scores or an automatic claim of improvement. Keep the `YZM5110/results/baselines` folder in Drive; share `metrics.json`, `history.csv`, and `run.json` for each model so we can review them together.
"""),
    code("""import json
import pandas as pd

rows = []
for mode in ('metadata', 'image', 'fusion'):
    result = json.loads((out_dir / mode / 'metrics.json').read_text())
    rows.append({'mode': mode, 'epoch': result['reported_epoch'],
                 'macro_f1': result['macro_f1'],
                 'balanced_accuracy': result['balanced_accuracy'],
                 'accuracy': result['accuracy'], 'roc_auc': result['roc_auc'],
                 'train_minutes': result['total_train_seconds'] / 60,
                 'peak_gpu_MB': result['peak_gpu_memory_mb']})
display(pd.DataFrame(rows))
print('Files saved under:', out_dir)
"""),
]

path = Path('notebooks/02_baselines_colab.ipynb')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)

