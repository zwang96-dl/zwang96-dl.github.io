/* KV Cache Stepper —— Lesson 7 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   KV Cache 是怎么随生成一步步增长的？prefill 与 decode 各往缓存里追加多少？
 *   为什么 decode 只需喂 1 个 token 就能得到和「重算整段」一样的结果？
 *
 * 镜像 mini_vllm/cache/kv_cache.py 与 transformer.forward(kv_cache=...)。
 */
(function () {
  "use strict";

  function buildFrames(promptLen, nNew) {
    var f = [{ cached: 0, fed: 0, phase: "start", newRows: 0,
               desc: "缓存为空。prompt 有 " + promptLen + " 个 token 等待 prefill。",
               code: "cache = KVCache(config)" }];
    // prefill：一次追加 promptLen 行
    f.push({ cached: promptLen, fed: promptLen, phase: "prefill", newRows: promptLen,
             desc: "Prefill：喂入整段 prompt（" + promptLen + " 个 token），一次性把它们的 K/V 追加进缓存，产出首 token。",
             code: "forward(prompt, cache)  →  cache.append(P 行)" });
    var cached = promptLen;
    for (var t = 1; t < nNew; t++) {
      cached += 1;
      f.push({ cached: cached, fed: 1, phase: "decode", newRows: 1,
               desc: "Decode 第 " + t + " 步：只喂入 1 个新 token；它的 K/V 追加进缓存（现有 " + cached +
                     " 行）。历史 K/V 直接复用，无需重算。",
               code: "forward([tok], [pos], cache)  →  cache.append(1 行)" });
    }
    f.push({ cached: cached, fed: 0, phase: "done", newRows: 0,
             desc: "全程：每步只算新 token 的 Q/K/V，历史 K/V 从缓存取。cached 与 naive 逐值一致（误差 0）。",
             code: "assert max_abs_diff(naive, cached) == 0" });
    return f;
  }

  function render(stage, f) {
    // 缓存格子：已有的 + 本步新增（高亮）
    var total = f.cached, newN = f.newRows, old = total - newN;
    var cells = '<div style="display:flex;flex-wrap:wrap;gap:4px;max-width:520px">';
    for (var i = 0; i < total; i++) {
      var isNew = i >= old;
      cells += '<div style="width:26px;height:26px;border-radius:5px;display:flex;align-items:center;' +
        'justify-content:center;font-family:var(--mono);font-size:.62rem;border:' +
        (isNew ? "2px solid var(--accent)" : "1px solid var(--border)") + ';background:' +
        (isNew ? "color-mix(in srgb, var(--accent) 25%, var(--bg))" : "var(--bg-code)") + '">' + i + '</div>';
    }
    cells += "</div>";
    var fedTag = f.fed > 0
      ? '<span style="color:var(--running)">本步喂入模型的 token 数 = ' + f.fed + '</span>'
      : '<span style="color:var(--fg-dim)">—</span>';
    stage.innerHTML =
      '<div style="font-size:.8rem;color:var(--fg-dim);margin-bottom:4px">阶段：<b>' + f.phase +
        '</b> · 缓存中的 K/V 行数 = ' + f.cached + ' · ' + fedTag + '</div>' + cells;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    window.Stepper.mount(root, { frames: buildFrames(cfg.promptLen || 5, cfg.nNew || 8), render: render });
  }
  window.KVCacheStepper = { mount: mount, buildFrames: buildFrames };
})();
