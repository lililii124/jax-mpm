"""Build a local review page for completed runs and available ParaView outputs."""

import argparse
import html
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

from collect_gallery import GROUPS, preview
from collect_results import SOURCES


def build(root, output):
    output.mkdir(parents=True, exist_ok=True)

    def link(path):
        return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/.")

    cards, pending = [], []
    selected = {relative: name for entries in GROUPS.values() for name, relative in entries}
    completed, largest = 0, 0
    for source in SOURCES:
        for summary_file in sorted((root / "outputs" / source).rglob("summary.json")):
            folder = summary_file.parent
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary.get("completed") is False or not (folder / "trajectory.npz").exists():
                continue
            completed += 1
            particles = sum(summary["particles"])
            largest = max(largest, particles)
            relative = folder.relative_to(root / "outputs").as_posix()
            name = selected.get(relative, str(summary["case"]))
            render_file = folder / "render.json"
            render = (
                json.loads(render_file.read_text(encoding="utf-8")) if render_file.exists() else {}
            )
            image, _, _ = preview(folder, name, render)
            if image.exists() and render.get("renderer") == "ParaView":
                thumbnail = f'<img loading="lazy" src="{link(image)}" alt="{html.escape(name)}">'
            else:
                thumbnail = '<div class="placeholder">渲染待完成</div>'
            video = folder / "simulation.mp4"
            if video.exists() and (folder / "video.json").exists():
                provenance = json.loads((folder / "video.json").read_text(encoding="utf-8"))
                if (
                    provenance.get("render_sha256")
                    == hashlib.sha256(render_file.read_bytes()).hexdigest()
                ):
                    thumbnail = (
                        f'<video controls preload="none" poster="{link(image)}">'
                        f'<source src="{link(video)}" type="video/mp4"></video>'
                    )
            links = []
            for filename, label in (
                ("summary.json", "参数与结果"),
                ("observables.csv", "响应曲线"),
                ("diagnostics.csv", "诊断数据"),
                ("particles.pvd", "ParaView"),
                ("scene.pvsm", "场景"),
            ):
                path = folder / filename
                if path.exists():
                    links.append(f'<a href="{link(path)}">{label}</a>')
            category = (
                "engineering"
                if source == "engineering"
                else "contact"
                if source == "contact-models"
                or name in ("oblique", "sliding", "rebound", "rubber-ball", "ball-beam")
                else "fluid"
                if name
                in ("dam-break", "sloshing", "fsi", "elastic-gate", "floating-block", "water-entry")
                else "material"
                if source in ("elastic_cantilever", "compression", "dense-materials")
                else "dynamics"
            )
            cards.append(
                (
                    particles,
                    f'<article data-category="{category}" data-name="{html.escape(relative)}">'
                    f'{thumbnail}<div class="body"><p class="path">{html.escape(relative)}</p>'
                    f'<h2>{html.escape(name)}</h2><p class="metrics"><b>{particles:,}</b> 物质点 · '
                    f'{summary["steps"]:,} 步 · {summary["simulation_time"]:.4g} s</p>'
                    f'<nav>{"".join(links)}</nav></div></article>',
                )
            )
        for status_file in sorted((root / "outputs" / source).rglob("status.json")):
            if (status_file.parent / "summary.json").exists():
                continue
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            label = status_file.parent.relative_to(root / "outputs").as_posix()
            pending.append(
                f'<li><a href="{link(status_file)}">{html.escape(label)}</a>'
                f' · 已保存帧 {status.get("frame", "—")}</li>'
            )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JAX MPM · 计算结果</title><style>
:root{font:15px/1.55 system-ui,sans-serif;color:#e6edf6;background:#101722}*{box-sizing:border-box}
body{max-width:1560px;margin:auto;padding:42px 28px}header{display:flex;justify-content:space-between;gap:28px;align-items:end}
h1{font-size:42px;letter-spacing:-1.5px;margin:0}header p{color:#a6b5c8;margin:10px 0}.stats{text-align:right;white-space:nowrap}
a{color:#91caff;text-decoration:none}a:hover{text-decoration:underline}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:30px 0 22px}
button,input{background:#1c293b;color:#e6edf6;border:1px solid #364458;border-radius:8px;padding:10px 14px;font:inherit}
button{cursor:pointer}button.active{background:#236ec0;border-color:#5798dd}input{margin-left:auto;min-width:220px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px}article{background:#1a2535;border:1px solid #2a394f;border-radius:14px;overflow:hidden}
img,video,.placeholder{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#f6f9fb}.placeholder{display:grid;place-items:center;color:#62748c}
.body{padding:18px 21px 22px}h2{font-size:22px;margin:4px 0 8px}.path{color:#95a7bd;font-size:11px;overflow-wrap:anywhere;margin:0}
.metrics{color:#b9c9dc;margin:0 0 18px}.metrics b{color:white}nav{display:flex;flex-wrap:wrap;gap:14px;font-size:13px}
footer{margin-top:36px;color:#a6b5c8}details{border-top:1px solid #344156;padding-top:20px}summary{cursor:pointer}li{margin:8px 0}
@media(max-width:600px){body{padding:24px 14px}header{display:block}.stats{text-align:left}h1{font-size:32px}input{width:100%;margin:8px 0 0}.grid{grid-template-columns:1fr}}
</style></head><body><header><div><h1>JAX MPM</h1><p>三维动力学 · 接触 · 材料响应 · 流固耦合</p>
<a href="https://github.com/lililii124/jax-mpm">GitHub</a></div><div class="stats">__STATS__</div></header>
<div class="toolbar"><button class="active" data-filter="all">全部</button><button data-filter="engineering">工程结构</button>
<button data-filter="contact">接触</button><button data-filter="fluid">流体与流固耦合</button><button data-filter="material">本构</button>
<button data-filter="dynamics">弹性动力学</button><input id="search" placeholder="查找算例" aria-label="查找算例"></div>
<main class="grid">__CARDS__</main><footer><p>图像由 ParaView 渲染，保留全部物质点。视频可直接播放；数据链接可用于独立检查。</p>
__PENDING__</footer><script>
let category='all';const search=document.querySelector('#search');function filter(){const q=search.value.toLowerCase();
document.querySelectorAll('article').forEach(c=>c.hidden=!((category==='all'||c.dataset.category===category)&&c.dataset.name.toLowerCase().includes(q)))}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{category=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(c=>c.classList.toggle('active',c===b));filter()});
search.oninput=filter;
</script></body></html>"""
    stats = f"<b>{completed}</b> 个完整运行<br>单例最大 <b>{largest:,}</b> 物质点"
    partial = (
        ("<details><summary>尚未完成的运行（未计入完整记录）</summary><ul>" + "".join(pending) + "</ul></details>")
        if pending
        else ""
    )
    document = (
        document.replace("__STATS__", stats)
        .replace("__CARDS__", "\n".join(card for _, card in sorted(cards, reverse=True)))
        .replace("__PENDING__", partial)
    )
    (output / "index.html").write_text(document, encoding="utf-8")
    print(f"{output / 'index.html'}: {completed} completed runs, largest {largest:,} particles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/review"))
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
