"""Generate the metadata-dropout improvement notebook from src/baseline.py."""
import hashlib
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(keepends=True)}


source = Path('src/baseline.py').read_text(encoding='utf-8')
sha = hashlib.sha256(source.encode('utf-8')).hexdigest()
embedded = source.split('if __name__ == "__main__":')[0]

cells = [
    md("""# YZM5110 — metadata-dropout improvement and control

Run this **after** the baseline notebook finishes and `MyDrive/YZM5110/results/baselines/fusion/final_model.pt` exists. The fixed plan is in [improvement_hypothesis.md](https://github.com/i-Emin/Multimodal-Skin-Cancer-Detection/blob/main/reports/improvement_hypothesis.md). This uses the same published train/validation manifests, with their known patient/lesion group overlap. No result is claimed before execution.

Only model change: set the full metadata vector to zero for 30% of **training** examples in a new fusion model. Compare both models on clean and fully missing metadata. Do not adjust `p` after seeing the scores. This notebook saves a resumable training checkpoint after each completed epoch in Drive; if Colab resets, rerun its cells from the top and the improvement training resumes from that checkpoint. The original baseline notebook does **not** have this safeguard.
"""),
    code("""from pathlib import Path
import torch
from google.colab import drive
import urllib.request, zipfile, shutil

if not torch.cuda.is_available():
    raise RuntimeError('Choose a GPU runtime.')
drive.mount('/content/drive')
data_dir = Path('/content/isic-dicm-17k')
baseline_dir = Path('/content/drive/MyDrive/YZM5110/results/baselines/fusion')
improvement_dir = Path('/content/drive/MyDrive/YZM5110/results/improvement')
if not (baseline_dir / 'final_model.pt').is_file():
    raise FileNotFoundError('Fusion baseline checkpoint missing. Finish notebook 02 first.')
if not (data_dir / 'images').is_dir() or not (data_dir / 'metadata.csv').is_file():
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
if len(list((data_dir / 'images').glob('*.jpg'))) != 17060:
    raise RuntimeError('Expected 17,060 JPG images. Check extraction before training.')
print('GPU:', torch.cuda.get_device_name(0))
print('Baseline checkpoint:', baseline_dir / 'final_model.pt')
"""),
    md("""## Training and evaluation code

Generated from `src/baseline.py`. Its exact SHA-256 is recorded in the new run configuration.
"""),
    code(embedded + f"\nSOURCE_SHA256 = '{sha}'\nprint('Source SHA-256:', SOURCE_SHA256)\n"),
    md("""## Train the single improvement

This is one additional eight-epoch fusion run. It saves a checkpoint and measured outputs to Drive.
"""),
    code("""import argparse

args = argparse.Namespace(mode='fusion', data_dir=str(data_dir),
                          output_dir=str(improvement_dir), epochs=8,
                          batch_size=32, lr=1e-4, seed=42, workers=2,
                          save_checkpoint=True, metadata_dropout_p=0.3,
                          resume=True)
if (improvement_dir / 'fusion' / 'final_model.pt').is_file() and (improvement_dir / 'fusion' / 'metrics.json').is_file():
    print('Improvement run already complete; using saved results.')
else:
    run(args)
"""),
    md("""## Matched clean/missing evaluation

Both checkpoints are evaluated with identical image transforms and train-fitted metadata encoding. `missing` means all encoded clinical features are set to zero. Results are saved as JSON and per-image predictions in Drive. The existing baseline is an appropriate `p=0` control for this intervention.
"""),
    code("""import json
import numpy as np
from torch.utils.data import DataLoader

train, valid, audit = load_manifests(data_dir / 'images')
encoder = MetadataEncoder().fit(train)
baseline_config = json.loads((baseline_dir / 'run.json').read_text())
if encoder.config() != baseline_config['metadata_preprocessing']:
    raise ValueError('Metadata preprocessing differs from baseline; stop and investigate.')
if baseline_config['seed'] != 42 or baseline_config['epochs'] != 8 or baseline_config['batch_size'] != 32 or baseline_config['lr'] != 1e-4:
    raise ValueError('Baseline settings differ from the pre-registered control.')

device = torch.device('cuda')
clean = encoder.transform(valid)
conditions = {'clean': clean, 'missing': np.zeros_like(clean)}
checkpoints = {'control': baseline_dir / 'final_model.pt',
               'dropout_0_3': improvement_dir / 'fusion' / 'final_model.pt'}
comparison = {'protocol': 'published_validation_with_group_overlap',
              'conditions': {}, 'source_sha256': SOURCE_SHA256}

for model_name, checkpoint in checkpoints.items():
    model = Baseline('fusion', encoder.dimension, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    for condition, encoded in conditions.items():
        dataset = LesionDataset(valid, encoded, data_dir / 'images', 'fusion', False)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        metrics, predictions = evaluate(model, loader, device)
        comparison['conditions'][f'{model_name}_{condition}'] = metrics
        predictions.to_csv(improvement_dir / f'{model_name}_{condition}_predictions.csv', index=False)
        print(model_name, condition, 'macro-F1', round(metrics['macro_f1'], 4),
              'balanced accuracy', round(metrics['balanced_accuracy'], 4),
              'ROC-AUC', round(metrics['roc_auc'], 4))

comparison['delta_macro_f1_missing'] = (
    comparison['conditions']['dropout_0_3_missing']['macro_f1'] -
    comparison['conditions']['control_missing']['macro_f1'])
comparison['delta_macro_f1_clean'] = (
    comparison['conditions']['dropout_0_3_clean']['macro_f1'] -
    comparison['conditions']['control_clean']['macro_f1'])
(improvement_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
print('Saved:', improvement_dir / 'comparison.json')
"""),
]

path = Path('notebooks/03_metadata_dropout_colab.ipynb')
path.write_text(json.dumps({"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)

