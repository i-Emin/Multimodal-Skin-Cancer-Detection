"""Generate the Colab starter notebook using only the Python standard library."""
import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md("""# YZM5110 — ISIC-DICM-17K: data audit and baseline preparation

This notebook starts with an **audit**, not a reported reproduction. Run it on Colab, inspect its printed columns and split diagnostics, and save the outputs before training. No numerical results are claimed here.

Source: [Ahammed et al., CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html). Official [ISIC-DICM-17K collection](https://api.isic-archive.com/doi/isic-dicm-17k/).

The paper used 15,098 training and 1,962 validation images, then evaluated on the ISIC 2020 test set through Kaggle. A new split of the downloadable bundle is **not** that published protocol. We must locate the original split IDs before claiming reproduction. The paper reports an ImageNet-pretrained EfficientNet-B3 multimodal AUC of 0.8851; this is an **authors' result**, not ours.
"""),
    code("""from pathlib import Path
import platform, sys, time, json
import pandas as pd
import numpy as np
import torch

print({'python': sys.version.split()[0], 'pandas': pd.__version__,
       'numpy': np.__version__, 'torch': torch.__version__,
       'cuda': torch.cuda.is_available(),
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
"""),
    md("""## Download the official bundle

The official complete ZIP is about 6.9 GB. The following cell downloads it into Colab's temporary storage. It may need substantial free disk space when extracted. Do not commit images or raw metadata to the public repository. Individual source images have different licenses (CC-0, CC-BY, CC-BY-NC); follow the attribution and noncommercial limits shown by ISIC.
"""),
    code("""import urllib.request, zipfile, shutil

DATA_DIR = Path('/content/isic-dicm-17k')
ZIP_PATH = Path('/content/isic-dicm-17k.zip')
URL = 'https://isic-archive.s3.amazonaws.com/dois/10.34970-233480/isic-dicm-17k.zip'
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not ZIP_PATH.exists():
    urllib.request.urlretrieve(URL, ZIP_PATH)
print('ZIP bytes:', ZIP_PATH.stat().st_size)
print('Free GB:', round(shutil.disk_usage('/content').free / 1e9, 2))
with zipfile.ZipFile(ZIP_PATH) as archive:
    names = archive.namelist()
    print('Entries:', len(names), 'Examples:', names[:12])
    bad = [n for n in names if Path(n).is_absolute() or '..' in Path(n).parts]
    if bad:
        raise ValueError(f'Unsafe archive paths: {bad[:5]}')
    if not any(DATA_DIR.iterdir()):
        archive.extractall(DATA_DIR)
"""),
    md("""## Inspect actual files and metadata

Do not assume the downloaded CSV has the same column names as the paper's broader ISIC Archive table. Inspect its schema and missingness first.
"""),
    code("""from collections import Counter

files = [p for p in DATA_DIR.rglob('*') if p.is_file()]
print('File count:', len(files))
print('Extensions:', Counter(p.suffix.lower() for p in files))
print('First paths:', [str(p.relative_to(DATA_DIR)) for p in files[:20]])
csv_paths = sorted(p for p in files if p.suffix.lower() == '.csv')
print('CSV files:', [str(p.relative_to(DATA_DIR)) for p in csv_paths])
for path in csv_paths:
    frame = pd.read_csv(path, low_memory=False)
    print('\\n', path.name, 'shape:', frame.shape)
    print('columns:', list(frame.columns))
    print('missing:', frame.isna().sum().sort_values(ascending=False).to_dict())
    print(frame.head(3).to_string(index=False))
"""),
    md("""## Validate labels, image mapping, and split identifiers

Set the four column variables **after** seeing the printed schema. Never put diagnosis, diagnosis confirmation, or label-derived fields among clinical inputs. If patient IDs are missing, report that limitation; image-level splitting alone cannot prove patient independence.
"""),
    code("""# Edit these only after inspecting the CSV above.
METADATA_CSV = None  # Example: csv_paths[0]
IMAGE_ID_COL = None
LABEL_COL = None
PATIENT_ID_COL = None
LESION_ID_COL = None

if METADATA_CSV is None or IMAGE_ID_COL is None or LABEL_COL is None:
    print('Set METADATA_CSV, IMAGE_ID_COL and LABEL_COL after auditing the real files.')
else:
    df = pd.read_csv(METADATA_CSV, low_memory=False)
    required = [IMAGE_ID_COL, LABEL_COL]
    assert all(c in df.columns for c in required)
    print('Labels:', df[LABEL_COL].value_counts(dropna=False).to_dict())
    print('Duplicate image IDs:', int(df[IMAGE_ID_COL].duplicated().sum()))
    for col in [PATIENT_ID_COL, LESION_ID_COL]:
        if col:
            assert col in df.columns, col
            print(col, 'missing:', int(df[col].isna().sum()),
                  'distinct:', df[col].nunique(dropna=True))
    image_paths = [p for p in files if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    image_by_stem = {p.stem: p for p in image_paths}
    ids = df[IMAGE_ID_COL].astype(str)
    print('Images found by filename stem:', int(ids.isin(image_by_stem).sum()), '/', len(ids))
    print('Unmatched IDs:', ids[~ids.isin(image_by_stem)].head(10).tolist())
"""),
    md("""## Split gate

Before model training: locate the original validation image IDs in the authors' repository or supplemental files and identify how ISIC 2020 test evaluation can be repeated. For a separate internal experiment, split by patient ID where complete and verify zero overlap; when IDs are incomplete, investigate lesion IDs and duplicate images. Fit imputers and encoders **only on training data**. Keep the test set untouched while choosing a method.

Next implementation stage: image-only pretrained CNN, metadata-only MLP, late-fusion model, and one matched-condition intervention, with macro-F1, balanced accuracy, accuracy, ROC-AUC, confusion matrix, per-class scores, timing, and saved configurations. Record deviations from the paper explicitly.
"""),
]

out = Path('notebooks/01_dataset_audit_colab.ipynb')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding='utf-8')
print(out)

