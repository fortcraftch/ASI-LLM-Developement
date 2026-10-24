"""Atomic per-file progress for future/resumed data scans."""
import json
import hashlib
from pathlib import Path
from recovery_v2.preflight import save


class ScanCheckpoint:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        path = self.directory/'identity.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Scan configuration or candidate hashes changed')
        else: save(path, identity)

    def path(self, name):
        return self.directory/(hashlib.sha256(name.encode()).hexdigest()+'.json')

    def read(self, name, source_sha256):
        path = self.path(name)
        if not path.exists(): return None
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['name'] != name or data['source_sha256'] != source_sha256:
            raise ValueError('Scanned file changed')
        payload = json.dumps(data['result'], sort_keys=True).encode()
        if hashlib.sha256(payload).hexdigest() != data['result_sha256']:
            raise ValueError('Corrupted scan checkpoint')
        return data['result']

    def write(self, name, source_sha256, result):
        if result.get('status') != 'complete':
            raise ValueError('Only completed file scans can be reused')
        previous = self.read(name, source_sha256)
        if previous is not None:
            if previous != result: raise ValueError('Conflicting completed scan')
            return
        save(self.path(name), {'name':name, 'source_sha256':source_sha256, 'result':result,
                              'result_sha256':hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})
