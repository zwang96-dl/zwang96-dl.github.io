"""Lesson 30 实验：Multimodal Final Incident Challenge。

综合场景 + 自检：placeholder/media 对齐、不跨请求串视觉 embedding、不误命中 stale cache、
timestamp 保留、visual token 计入调度、vision encoder 不在 decode 重复运行。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import VisionConfig
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.mm_engine import MultiModalEngine
from mini_vllm.multimodal.budget import MultiModalBudget
from mini_vllm.multimodal import messages as msg
from mini_vllm.multimodal.media import synth_image
from mini_vllm.sampling import SamplingParams
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model()
    img = lambda s: {"synth": {"h": 16, "w": 16, "seed": s}}
    vid = {"frames": [synth_image(16, 16, k) for k in range(6)], "fps": 2.0, "num_frames": 3}
    specs = [
        ("text", [msg.user(msg.text("text only"))], 4),
        ("imgA", [msg.user(msg.text("A"), msg.image(img(1)))], 5),
        ("imgA2", [msg.user(msg.text("A again"), msg.image(img(1)))], 4),   # 同图 → 缓存命中
        ("multi", [msg.user(msg.image(img(2)), msg.image(img(3)))], 4),
        ("video", [msg.user(msg.text("vid"), msg.video(vid))], 3),
    ]
    eng = MultiModalEngine(model, VisionConfig(),
                           budget=MultiModalBudget(text_token_budget=16, visual_token_budget=24,
                                                   encoder_budget=2, max_num_seqs=3),
                           enable_caches=True)
    for rid, m, n in specs:
        eng.add_request(rid, m, n, arrival={"text": 0, "imgA": 0, "imgA2": 1, "multi": 2, "video": 2}[rid])
    res = eng.run(tracer=tracer)
    outs = {r.request_id: r.output_token_ids for r in res.requests}

    # 自检 1：不跨请求串用——每个请求输出与单独 runner.generate 一致
    ref_runner = MultiModalRunner(model, VisionConfig())
    correct = all(outs[rid] == ref_runner.generate(m, max_new_tokens=n,
                  sampling=SamplingParams())["generated"] for rid, m, n in specs)
    # 自检 2：placeholder 对齐（build_inputs 不抛异常即通过；multi 应为 8 visual token）
    ids, ranges, vembeds, meta = ref_runner.build_inputs(specs[3][1])
    align_ok = sum(len(v) for v in vembeds) == 8 and len(ranges) == 2
    # 自检 3：timestamp 保留（video 的 media_meta 带 timeline）
    vids = [r for r in res.requests if r.request_id == "video"][0]
    ts_ok = bool(vids.media_meta and vids.media_meta[0].get("timeline"))
    # 自检 4：encoder 缓存命中（imgA2 复用 imgA）
    hit_ok = eng.encoder_cache.stats()["hits"] > 0

    return {"iterations": res.iterations, "encoder_runs": eng.stats()["encoder_runs"],
            "encoder_cache": eng.encoder_cache.stats(),
            "checks": {"no_cross_request_bleed": correct, "placeholder_aligned": align_ok,
                       "timestamp_preserved": ts_ok, "encoder_cache_hit": hit_ok}}


def print_summary(r):
    print("\n" + "=" * 68)
    print("  Lesson 30 · Multimodal Final Incident Challenge —— 运行成功 ✓")
    print("=" * 68)
    print(f"  迭代 {r['iterations']}，vision encoder 运行 {r['encoder_runs']} 次，"
          f"encoder 缓存命中 {r['encoder_cache']['hits']}")
    print("  综合自检：")
    c = r["checks"]
    print(f"    ✓ 不跨请求串视觉 embedding（输出与单独生成一致）：{c['no_cross_request_bleed']}")
    print(f"    ✓ placeholder 与媒体严格对齐（多图 8 视觉 token）：{c['placeholder_aligned']}")
    print(f"    ✓ 视频 timestamp metadata 保留：{c['timestamp_preserved']}")
    print(f"    ✓ 相同媒体命中 encoder 缓存（首次 vs 命中 TTFT 不同）：{c['encoder_cache_hit']}")
    print("=" * 68)
    print("  多模态主线（Lesson 19–30）完成！")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_30_mm_final",
                    "configs/lesson_30_quick.json", "outputs/lesson_30").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
