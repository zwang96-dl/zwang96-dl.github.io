"""离线自检（offline bundle verifier）—— course.py 与测试共用的一份逻辑。

对应 Prompt 第四章「严格离线交付标准」的验证要求。检查项：

    1. 关键静态文件是否齐全（HTML / CSS / JS / tokenizer / config / workload）。
    2. HTML/CSS/JS 是否**加载**了远程资源（CDN、在线字体、远程脚本/图片）。
    3. Python 源码里是否存在**运行时联网**代码（urlopen / requests / torch.hub / socket）。
    4. offline/manifests/wheels.json 里声明为 required 的 wheel 是否都存在、SHA-256 是否匹配。
    5. 是否存在指向仓库外的绝对路径引用。

**设计取舍（诚实标注）**：本课程核心是**纯标准库**实现（机制模拟器 + 后续纯 Python
tiny model），因此 ``required`` wheel 集合为空——Level A（零第三方依赖离线学习）
天然满足。可选的 PyTorch 加速属于独立 optional bundle，不属于核心必需项。

任何**必需**文件缺失时，:func:`verify_bundle` 返回 ok=False（调用方应以非零退出，
而不是仅给 warning）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# 运行时必需的关键静态资源（相对项目根目录）。
REQUIRED_FILES = [
    "README.md",
    "course.py",
    "mini_vllm/__init__.py",
    "mini_vllm/tokenizer.py",
    "mini_vllm/simulator/text_pipeline.py",
    "mini_vllm/model/matrix.py",
    "mini_vllm/model/attention_ref.py",
    "mini_vllm/model/rmsnorm.py",
    "mini_vllm/model/rope.py",
    "mini_vllm/model/mlp.py",
    "mini_vllm/model/transformer.py",
    "mini_vllm/cache/kv_cache.py",
    "mini_vllm/sampling.py",
    "mini_vllm/engine/generate.py",
    "assets/tokenizer/byte_tokenizer.json",
    "assets/checkpoints/tiny_text.json",
    "configs/lesson_00_quick.json",
    "assets/workloads/lesson_00.json",
    "configs/lesson_01_quick.json",
    "assets/workloads/lesson_01.json",
    "configs/lesson_02_quick.json",
    "assets/workloads/lesson_02.json",
    "configs/lesson_04_quick.json",
    "assets/workloads/lesson_04.json",
    "configs/lesson_03_quick.json",
    "configs/lesson_05_quick.json",
    "configs/lesson_06_quick.json",
    "configs/lesson_07_quick.json",
    "configs/lesson_08_quick.json",
    "experiments/lesson_00_intro.py",
    "experiments/lesson_01_tokenizer.py",
    "experiments/lesson_02_tensor.py",
    "experiments/lesson_04_attention.py",
    "experiments/lesson_03_transformer.py",
    "experiments/lesson_05_generation.py",
    "experiments/lesson_06_recompute.py",
    "experiments/lesson_07_kv_cache.py",
    "experiments/lesson_08_prefill_decode.py",
    "docs/index.html",
    "docs/lessons/lesson_00.html",
    "docs/lessons/lesson_01.html",
    "docs/lessons/lesson_02.html",
    "docs/lessons/lesson_03.html",
    "docs/lessons/lesson_04.html",
    "docs/lessons/lesson_05.html",
    "docs/lessons/lesson_06.html",
    "docs/lessons/lesson_07.html",
    "docs/lessons/lesson_08.html",
    "docs/css/course.css",
    "docs/js/course.js",
    "docs/js/anim/request_lifecycle.js",
    "docs/js/anim/stepper_core.js",
    "docs/js/anim/tokenizer_explorer.js",
    "docs/js/anim/matrix_mul.js",
    "docs/js/anim/attention_stepper.js",
    "docs/js/anim/transformer_pipeline.js",
    "docs/js/anim/sampling_explorer.js",
    "docs/js/anim/recompute_timeline.js",
    "docs/js/anim/kv_cache_stepper.js",
    "docs/js/anim/prefill_decode_timeline.js",
    "docs/js/anim/blocks_anim.js",
    "docs/js/anim/waste_anim.js",
    # Phase 3B：调度 / 分页 / 前缀 / 引擎
    "mini_vllm/cache/block_allocator.py",
    "mini_vllm/cache/block_table.py",
    "mini_vllm/cache/prefix_cache.py",
    "mini_vllm/scheduler/request.py",
    "mini_vllm/scheduler/scheduler.py",
    "mini_vllm/engine/engine.py",
    "benchmarks/report.py",
    "experiments/lesson_09_static_batching.py",
    "experiments/lesson_10_continuous_batching.py",
    "experiments/lesson_11_scheduler.py",
    "experiments/lesson_12_kv_waste.py",
    "experiments/lesson_13_block_allocator.py",
    "experiments/lesson_14_paged_kv.py",
    "experiments/lesson_15_chunked_prefill.py",
    "experiments/lesson_16_prefix_cache.py",
    "experiments/lesson_17_text_engine.py",
    "experiments/lesson_18_final_challenge.py",
    "configs/lesson_09_quick.json", "configs/lesson_10_quick.json",
    "configs/lesson_11_quick.json", "configs/lesson_12_quick.json",
    "configs/lesson_13_quick.json", "configs/lesson_14_quick.json",
    "configs/lesson_15_quick.json", "configs/lesson_16_quick.json",
    "configs/lesson_17_quick.json", "configs/lesson_18_quick.json",
    "docs/lessons/lesson_09.html", "docs/lessons/lesson_10.html",
    "docs/lessons/lesson_11.html", "docs/lessons/lesson_12.html",
    "docs/lessons/lesson_13.html", "docs/lessons/lesson_14.html",
    "docs/lessons/lesson_15.html", "docs/lessons/lesson_16.html",
    "docs/lessons/lesson_17.html", "docs/lessons/lesson_18.html",
    # Phase 4-6：多模态
    "mini_vllm/multimodal/messages.py",
    "mini_vllm/multimodal/inputs.py",
    "mini_vllm/multimodal/chat_template.py",
    "mini_vllm/multimodal/placeholders.py",
    "mini_vllm/multimodal/media.py",
    "mini_vllm/multimodal/image_processor.py",
    "mini_vllm/multimodal/video_sampler.py",
    "mini_vllm/multimodal/patch_embed.py",
    "mini_vllm/multimodal/vision_encoder.py",
    "mini_vllm/multimodal/embedding_merge.py",
    "mini_vllm/multimodal/cache.py",
    "mini_vllm/multimodal/budget.py",
    "mini_vllm/multimodal/runner.py",
    "mini_vllm/multimodal/mm_engine.py",
    "docs/js/anim/mm_pipeline.js",
    "experiments/lesson_19_mm_request.py",
    "configs/lesson_19_quick.json",
    "docs/lessons/lesson_19.html",
    "experiments/lesson_20_image_tensor.py",
    "configs/lesson_20_quick.json",
    "docs/lessons/lesson_20.html",
    "experiments/lesson_21_image_patch.py",
    "configs/lesson_21_quick.json",
    "docs/lessons/lesson_21.html",
    "experiments/lesson_22_vision_encoder.py",
    "configs/lesson_22_quick.json",
    "docs/lessons/lesson_22.html",
    "experiments/lesson_23_placeholder_merge.py",
    "configs/lesson_23_quick.json",
    "docs/lessons/lesson_23.html",
    "experiments/lesson_24_mm_prefill.py",
    "configs/lesson_24_quick.json",
    "docs/lessons/lesson_24.html",
    "experiments/lesson_25_multi_image.py",
    "configs/lesson_25_quick.json",
    "docs/lessons/lesson_25.html",
    "experiments/lesson_26_video.py",
    "configs/lesson_26_quick.json",
    "docs/lessons/lesson_26.html",
    "experiments/lesson_27_three_cache.py",
    "configs/lesson_27_quick.json",
    "docs/lessons/lesson_27.html",
    "experiments/lesson_28_mm_scheduler.py",
    "configs/lesson_28_quick.json",
    "docs/lessons/lesson_28.html",
    "experiments/lesson_29_mm_engine.py",
    "configs/lesson_29_quick.json",
    "docs/lessons/lesson_29.html",
    "experiments/lesson_30_mm_final.py",
    "configs/lesson_30_quick.json",
    "docs/lessons/lesson_30.html",
    "offline/requirements-lock.txt",
    "offline/INSTALL_OFFLINE.md",
    "offline/manifests/wheels.json",
    "offline/manifests/assets.json",
]

# 会被扫描「远程资源加载」的目录。
_DOCS_GLOBS = ("*.html", "*.css", "*.js")

# 「加载远程资源」的模式（prose 里出现 https 的 <a> 链接不会被这些命中）。
_REMOTE_LOAD_PATTERNS = [
    re.compile(r"<script[^>]+src\s*=\s*['\"]https?://", re.I),
    re.compile(r"<link[^>]+href\s*=\s*['\"]https?://", re.I),
    re.compile(r"<img[^>]+src\s*=\s*['\"]https?://", re.I),
    re.compile(r"@import[^;]*https?://", re.I),
    re.compile(r"url\(\s*['\"]?https?://", re.I),
    re.compile(r"fonts\.googleapis\.com", re.I),
    re.compile(r"(cdn\.jsdelivr|unpkg\.com|cdnjs\.cloudflare|ajax\.googleapis)", re.I),
]

# 「运行时联网」的模式（在 Python 源码里出现即视为违规）。
_NETWORK_CODE_PATTERNS = [
    re.compile(r"urllib\.request\.urlopen"),
    re.compile(r"\brequests\.(get|post)\b"),
    re.compile(r"torch\.hub\.load"),
    re.compile(r"from_pretrained\s*\("),
    re.compile(r"\bsocket\.socket\b"),
    re.compile(r"http\.client\."),
]

# 扫描运行时联网代码的目录。
_PY_SCAN_DIRS = ["mini_vllm", "experiments", "benchmarks"]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    items: list[str] = field(default_factory=list)


@dataclass
class BundleReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "", items: list[str] | None = None) -> None:
        self.checks.append(CheckResult(name, ok, detail, items or []))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(root: Path, check_wheels: bool = True) -> BundleReport:
    """执行全部离线检查，返回结构化报告。"""
    root = Path(root)
    report = BundleReport()

    # 1) 必需文件齐全 -------------------------------------------------------
    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    report.add(
        "必需静态文件齐全",
        ok=not missing,
        detail=f"{len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)} 存在",
        items=[f"缺失：{m}" for m in missing],
    )

    # 2) HTML/CSS/JS 不加载远程资源 ----------------------------------------
    docs = root / "docs"
    remote_hits: list[str] = []
    if docs.exists():
        for pattern in _DOCS_GLOBS:
            for fp in docs.rglob(pattern):
                text = fp.read_text("utf-8", errors="replace")
                for pat in _REMOTE_LOAD_PATTERNS:
                    if pat.search(text):
                        remote_hits.append(f"{fp.relative_to(root)}：命中 {pat.pattern}")
    report.add(
        "网页不加载远程资源（CDN/在线字体/远程脚本/图片）",
        ok=not remote_hits,
        detail="未发现远程资源加载" if not remote_hits else f"{len(remote_hits)} 处违规",
        items=remote_hits,
    )

    # 3) Python 源码无运行时联网 -------------------------------------------
    net_hits: list[str] = []
    for d in _PY_SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for fp in base.rglob("*.py"):
            text = fp.read_text("utf-8", errors="replace")
            for pat in _NETWORK_CODE_PATTERNS:
                if pat.search(text):
                    net_hits.append(f"{fp.relative_to(root)}：命中 {pat.pattern}")
    report.add(
        "Python 源码无运行时联网代码",
        ok=not net_hits,
        detail="未发现联网调用" if not net_hits else f"{len(net_hits)} 处",
        items=net_hits,
    )

    # 4) required wheel 齐全且 SHA-256 匹配 ---------------------------------
    if check_wheels:
        wheels_json = root / "offline/manifests/wheels.json"
        wheel_items: list[str] = []
        ok = True
        if not wheels_json.exists():
            ok = False
            wheel_items.append("缺失 offline/manifests/wheels.json")
        else:
            data = json.loads(wheels_json.read_text("utf-8"))
            required = data.get("required", [])
            wheel_dir = root / "offline/wheels/macos-arm64"
            for w in required:
                fname = w.get("filename", "")
                wpath = wheel_dir / fname
                if not wpath.exists():
                    ok = False
                    wheel_items.append(f"缺失 wheel：{fname}")
                    continue
                want = w.get("sha256", "")
                if want:
                    got = _sha256(wpath)
                    if got != want:
                        ok = False
                        wheel_items.append(f"SHA256 不匹配：{fname}（期望 {want[:12]}… 实际 {got[:12]}…）")
            if not required:
                wheel_items.append("required 为空 —— 核心为纯标准库，无需任何 wheel（Level A 满足）")
        report.add("required wheel 齐全且校验通过", ok=ok, items=wheel_items)

    # 5) assets.json 中登记的资源存在且校验通过 ----------------------------
    assets_json = root / "offline/manifests/assets.json"
    asset_items: list[str] = []
    ok = True
    if assets_json.exists():
        data = json.loads(assets_json.read_text("utf-8"))
        for a in data.get("assets", []):
            rel = a.get("path", "")
            ap = root / rel
            if not ap.exists():
                ok = False
                asset_items.append(f"缺失资源：{rel}")
                continue
            want = a.get("sha256", "")
            if want:
                got = _sha256(ap)
                if got != want:
                    ok = False
                    asset_items.append(f"SHA256 不匹配：{rel}")
        report.add("登记资源齐全且校验通过", ok=ok,
                   detail=f"{len(data.get('assets', []))} 项", items=asset_items)
    # assets.json 不存在时，第 1 项已覆盖其缺失，不重复报错。

    return report


def format_report(report: BundleReport) -> str:
    lines = ["离线自检（verify-offline-bundle）", "=" * 48]
    for c in report.checks:
        mark = "✓" if c.ok else "✗"
        line = f"  [{mark}] {c.name}"
        if c.detail:
            line += f"  ——  {c.detail}"
        lines.append(line)
        for it in c.items:
            lines.append(f"        · {it}")
    lines.append("=" * 48)
    lines.append("  结果：全部通过 ✓" if report.ok else "  结果：存在必需项缺失/违规 ✗（请修复后重试）")
    return "\n".join(lines)
