"""Encode and independently decode every real rendered frame."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, GifImagePlugin

parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()
root = Path(args.output)
render = json.loads((root/'render.json').read_text())
files = sorted((root/'png').glob('*.png'))
if len(files) != render['frames'] or (not args.smoke and (len(files) != 13 or render['smoke'])):
    raise RuntimeError('Exact complete render series required')
frames = []
for path, record in zip(files, render['records']):
    if path.name != record['png'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256'] or not record['fetched_exact']:
        raise RuntimeError('PNG or geometry receipt differs')
    with Image.open(path) as image:
        if image.size != (400, 400):
            raise RuntimeError('Expected 400 square frames')
        frames.append(image.convert('RGB').quantize(colors=256))
target = root/'phase-field-bending-medium1200.gif'
with target.open('wb') as stream:
    header, _ = GifImagePlugin.getheader(frames[0], info={'loop': 0})
    for block in header:
        stream.write(block)
    for frame in frames:
        for block in GifImagePlugin.getdata(frame, duration=100, disposal=2, include_color_table=True):
            stream.write(block)
    stream.write(b';')
hashes = []
with Image.open(target) as image:
    if image.n_frames != len(frames) or image.size != (400, 400):
        raise RuntimeError('Decoded GIF frame count or dimensions differ')
    for index, expected in enumerate(frames):
        image.seek(index)
        decoded = image.convert('RGB').tobytes()
        if decoded != expected.convert('RGB').tobytes():
            raise RuntimeError('Decoded GIF frame differs from its exact quantized PNG')
        hashes.append(hashlib.sha256(decoded).hexdigest())
if len(set(hashes)) < 2:
    raise RuntimeError('Static decoded GIF rejected')
(root/'gif-checks.json').write_text(json.dumps(dict(frames=len(frames),
    decoded_exact=True, unique_decoded_frames=len(set(hashes)), decoded_hashes=hashes,
    gif_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), smoke=args.smoke), indent=2))
