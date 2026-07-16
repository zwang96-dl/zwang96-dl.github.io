/* Prefill / Decode Timeline —— Lesson 8 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   一次请求的时间线长什么样？prefill 与 decode 的成本差别在哪？TTFT/TPOT 指的是什么？
 *
 * 用「处理的 token 数」作为成本代理（prefill=prompt_len，decode=1），镜像 Lesson 8 实验。
 */
(function () {
  "use strict";

  function buildFrames(promptLen, nDecode) {
    var f = [{ blocks: [], desc: "请求到达。prompt 有 " + promptLen + " 个 token。",
               code: "generate_cached(prompt, n)" }];
    var blocks = [{ kind: "prefill", cost: promptLen, label: "Prefill (" + promptLen + ")" }];
    f.push({ blocks: blocks.slice(),
             desc: "Prefill：一次处理整段 prompt（成本 ∝ " + promptLen + "），产出首 token。这段耗时就是 TTFT。",
             code: "TTFT = prefill 耗时" });
    for (var t = 1; t <= nDecode; t++) {
      blocks = blocks.concat([{ kind: "decode", cost: 1, label: "d" + t }]);
      f.push({ blocks: blocks.slice(),
               desc: "Decode 第 " + t + " 步：只处理 1 个 token（成本小且稳定）。相邻两个 token 的间隔就是 ITL。",
               code: "TPOT = decode 步平均耗时" });
    }
    f.push({ blocks: blocks.slice(),
             desc: "整条时间线：一个宽的 prefill 块 + 若干窄的 decode 块。prompt 越长，prefill 块越宽、TTFT 越大。",
             code: "prompt-heavy → 大 TTFT；decode-heavy → TTFT 小、由 TPOT 主导" });
    return f;
  }

  function render(stage, f) {
    var html = '<div style="display:flex;align-items:flex-end;gap:3px;height:70px;padding:6px 0">';
    f.blocks.forEach(function (b) {
      var w = b.kind === "prefill" ? Math.max(24, b.cost * 8) : 16;
      var bg = b.kind === "prefill" ? "var(--prefill)" : "var(--decode)";
      var border = b.kind === "prefill" ? "3px double var(--prefill)" : "2px dotted var(--decode)";
      html += '<div title="' + b.label + '" style="width:' + w + 'px;height:46px;background:' +
        'color-mix(in srgb,' + bg + ' 25%, var(--bg));border:' + border +
        ';border-radius:6px;display:flex;align-items:center;justify-content:center;' +
        'font-size:.7rem;font-family:var(--mono)">' + b.label + '</div>';
    });
    html += "</div>";
    html += '<div class="legend" style="padding:4px 0">' +
      '<span class="item"><span class="swatch prefill"></span>Prefill（宽 = prompt 越长越贵 → TTFT）</span>' +
      '<span class="item"><span class="swatch decode"></span>Decode（每步 1 token → TPOT/ITL）</span></div>';
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    window.Stepper.mount(root, { frames: buildFrames(cfg.promptLen || 8, cfg.nDecode || 8), render: render });
  }
  window.PrefillDecodeTimeline = { mount: mount, buildFrames: buildFrames };
})();
