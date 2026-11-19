"""Streaming audit and bounded per-class reservoir sampling from CIC-IDS2017 CSV ZIP."""
from collections import Counter
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ROOT, config

# Do not include identifiers or the target in the predictors.
DROP = {"label", "flow id", "source ip", "destination ip", "src ip", "dst ip", "timestamp", "source port", "destination port"}

def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_samples(zip_path, limit=30000, chunk_size=25000, seed=42):
    """Uniform reservoir sample within each binary class across all files.
    The global class-specific reservoir avoids first-N selection and bounds RAM.
    Returns (sample, audit). The audit counts are before cleaning.
    """
    rng = np.random.default_rng(seed)
    reservoirs = {0: pd.DataFrame(), 1: pd.DataFrame()}; seen = Counter(); raw_labels = Counter()
    file_counts = {}; columns = None
    with zipfile.ZipFile(zip_path) as z:
        files = sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
        if not files: raise ValueError('No CSV files found in ZIP')
        for filename in files:
            count = 0
            with z.open(filename) as stream:
                for chunk in pd.read_csv(stream, chunksize=chunk_size, low_memory=False, on_bad_lines='error'):
                    chunk = normalize_columns(chunk)
                    if 'Label' not in chunk: raise ValueError(f'Label missing in {filename}')
                    if columns is None: columns = list(chunk.columns)
                    if set(chunk.columns) != set(columns): raise ValueError(f'Column mismatch in {filename}')
                    labels = chunk['Label'].astype(str).str.strip()
                    raw_labels.update(labels.value_counts().to_dict())
                    y = (labels.str.upper() != 'BENIGN').astype(np.int8).to_numpy()
                    count += len(chunk)
                    for cls in (0,1):
                        mask = y == cls; n = int(mask.sum()); seen[cls] += n
                        if n == 0: continue
                        part = chunk.loc[mask].copy()
                        part['_capture_file'] = filename
                        part['_random_priority'] = rng.random(n)
                        reservoirs[cls] = pd.concat([reservoirs[cls],part],ignore_index=True)
                        if len(reservoirs[cls]) > limit:
                            reservoirs[cls] = reservoirs[cls].nsmallest(limit,'_random_priority').reset_index(drop=True)
            file_counts[filename] = count
    sample = pd.concat([reservoirs[0],reservoirs[1]],ignore_index=True).drop(columns='_random_priority')
    sample = normalize_columns(sample)
    audit = {'raw_rows': int(sum(seen.values())), 'raw_binary_counts': {str(k):int(v) for k,v in seen.items()}, 'raw_label_counts': {str(k):int(v) for k,v in raw_labels.items()}, 'file_rows': file_counts, 'sample_rows': len(sample), 'sampling': 'uniform random-priority sample per binary class; before cleaning', 'seed':seed}
    return sample, audit

def prepare(sample, seed=42, validation_fraction=.15, test_fraction=.15):
    sample = normalize_columns(sample.copy())
    y = (sample['Label'].astype(str).str.strip().str.upper() != 'BENIGN').astype(int)
    feature_cols = [c for c in sample if c.lower() not in DROP and not c.startswith('_')]
    x = sample[feature_cols].apply(pd.to_numeric, errors='coerce')
    x = x.replace([np.inf, -np.inf], np.nan)
    # Remove exact duplicates across features AND label before splitting.
    combined = x.copy(); combined['_target'] = y.to_numpy()
    duplicates = int(combined.duplicated().sum())
    keep = ~combined.duplicated()
    x = x.loc[keep].reset_index(drop=True); y = y.loc[keep].reset_index(drop=True)
    if y.nunique() != 2: raise ValueError('Both classes required after cleaning')
    if y.value_counts().min() < 3: raise ValueError('Too few minority-class rows')
    idx_trainval, idx_test = train_test_split(np.arange(len(y)), test_size=test_fraction, stratify=y, random_state=seed)
    relative_val = validation_fraction / (1-test_fraction)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=relative_val, stratify=y.iloc[idx_trainval], random_state=seed)
    splits = {k:(x.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)) for k,idx in [('train',idx_train),('val',idx_val),('test',idx_test)]}
    return splits, {'feature_columns':feature_cols,'removed_exact_duplicates':duplicates,'split_counts':{k:{str(c):int(n) for c,n in yy.value_counts().items()} for k,(_,yy) in splits.items()},'split_method':'stratified random split after exact duplicate removal; not independent capture-day generalization'}

def run_audit(zip_path, out_dir=None, cfg=None):
    cfg = cfg or config(); out_dir = Path(out_dir or ROOT/'artifacts'); out_dir.mkdir(parents=True, exist_ok=True)
    sample, audit = load_samples(zip_path, cfg['max_rows_per_class'],cfg['chunk_size'],cfg['seed'])
    splits, prep = prepare(sample,cfg['seed'],cfg['validation_fraction'],cfg['test_fraction'])
    for name,(x,y) in splits.items():
        x.assign(_target=y).to_pickle(out_dir/f'{name}.pkl')
    audit.update(prep)
    (out_dir/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit
