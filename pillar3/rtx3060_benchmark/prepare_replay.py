"""Prepare a writable replay copy using private frames; no inference or network.

python prepare_replay.py --frames-dir /private/images --output /new/workdir
The output must not exist. JPEGs are referenced directly, never copied.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); source=Path(__file__).resolve().parent
    if a.output.exists(): p.error('output already exists')
    if not a.frames_dir.is_dir(): p.error('private images root does not exist')
    shutil.copytree(source,a.output)
    common=a.output/'p3common.py'; text=common.read_text(encoding='utf-8')
    old='p = BUNDLE / "frames" / acq / f"{sec:06d}.jpg"'
    if old not in text: raise ValueError('archived frame lookup changed')
    text=text.replace(old,f'p = Path({str(a.frames_dir.resolve())!r}) / acq / f"{{sec:06d}}.jpg"')
    common.write_text(text,encoding='utf-8')
    (a.output/'results').mkdir(exist_ok=True)
    with (a.output/'SHA256SUMS.txt').open('w',encoding='utf-8',newline='\n') as f:
        for file in sorted(a.output.rglob('*')):
            if file.is_file() and file.name!='SHA256SUMS.txt': f.write(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+file.relative_to(a.output).as_posix()+'\n')
    print('Prepared only. No server, model or benchmark started. Validate the private frame hashes before replay.')

if __name__=='__main__': main()
