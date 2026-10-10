"""Shared artifact helpers for offline research reports."""
import hashlib
import json
from pathlib import Path


def read_json(path, provenance):
    raw=Path(path).read_bytes()
    provenance[str(path)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save_figure(fig, output, name, dpi=160):
    import matplotlib.pyplot as plt
    for extension in ('png','pdf'):
        fig.savefig(Path(output)/f'{name}.{extension}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
