"""Lesson 26 实验：视频如何变成模型输入（抽帧策略、timestamp、时间语义）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result
from mini_vllm.multimodal.media import synth_image, VideoFrame
from mini_vllm.multimodal.video_sampler import VideoFrameSampler
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    n = cfg.get("num_source_frames", 10)
    fps = cfg.get("source_fps", 2.0)
    frames = [VideoFrame(pixels=synth_image(16, 16, k), frame_index=k, timestamp_seconds=k / fps)
              for k in range(n)]
    sampler = VideoFrameSampler()
    strategies = {
        "uniform(4)": sampler.sample(frames, "uniform", num_frames=4, source_fps=fps),
        "fixed_fps(1)": sampler.sample(frames, "fixed_fps", fps=1.0, source_fps=fps),
        "head(3)": sampler.sample(frames, "head", num_frames=3),
        "tail(3)": sampler.sample(frames, "tail", num_frames=3),
    }
    out = {}
    with tracer.section("frame sampling"):
        for name, s in strategies.items():
            tl = sampler.timeline(s)
            out[name] = {"count": len(s), "frame_indices": [f["frame_index"] for f in tl],
                         "timestamps": [f["timestamp"] for f in tl]}
            tracer.event(name, frames=[f["frame_index"] for f in tl])
    return {"source_frames": n, "source_fps": fps, "strategies": out}


def print_summary(r):
    print("\n" + "=" * 68)
    print("  Lesson 26 · 视频如何变成模型输入 —— 运行成功 ✓")
    print("=" * 68)
    print(f"  源视频：{r['source_frames']} 帧 @ {r['source_fps']} fps")
    print(f"  {'策略':<16}{'帧数':>5}  帧号 / 时间戳(s)")
    for name, d in r["strategies"].items():
        print(f"  {name:<16}{d['count']:>5}  {d['frame_indices']} / {d['timestamps']}")
    print("-" * 68)
    print("  证据：不同策略在覆盖度与成本间权衡；每帧都带 frame_index 与 timestamp。")
    print("        timestamp 是 processor metadata——不显式喂进 prompt，LLM 不会自动理解它。")
    print("=" * 68)
    print("  下一步：python3 course.py check 26   或   Lesson 27 三层缓存。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_26_video",
                    "configs/lesson_26_quick.json", "outputs/lesson_26").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
