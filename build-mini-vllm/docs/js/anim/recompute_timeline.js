/* Recompute Timeline —— Lesson 6 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   朴素生成为什么越来越慢？naive 与 cached 每步各处理多少 token？
 *
 * 镜像 mini_vllm/engine/generate.py 的 processed_token_curves()。
 */
(function () {
  "use strict";

  function buildFrames(promptLen, nNew) {
    var f = [{ step: -1, desc: "起点：prompt 有 " + promptLen + " 个 token。下面比较逐步「处理的 token 数」。",
               code: "generate_naive() vs generate_cached()", naive: [], cached: [], nTot: 0, cTot: 0 }];
    var nTot = 0, cTot = 0, naive = [], cached = [];
    for (var t = 0; t < nNew; t++) {
      var nP = promptLen + t;                 // naive 每步重算整段
      var cP = (t === 0) ? promptLen : 1;     // cached 只在 prefill 处理整段
      nTot += nP; cTot += cP;
      naive = naive.concat([nP]); cached = cached.concat([cP]);
      f.push({
        step: t,
        desc: "第 " + t + " 步：naive 处理 " + nP + " 个 token（重算整段前缀），cached 只处理 " + cP + " 个。",
        code: t === 0 ? "prefill：两者都处理整段 prompt" : "decode：naive 重算 " + nP + "，cached 只算 1",
        naive: naive.slice(), cached: cached.slice(), nTot: nTot, cTot: cTot
      });
    }
    var ratio = (nTot / cTot).toFixed(1);
    f.push({ step: nNew, desc: "累计：naive=" + nTot + " vs cached=" + cTot + " —— naive 多做 " + ratio +
             "×。这就是 KV Cache 要消除的浪费（O(n²) → ~O(n)）。",
             code: "naive_total=" + nTot + ", cached_total=" + cTot, naive: naive, cached: cached, nTot: nTot, cTot: cTot });
    return f;
  }

  function bars(title, arr, color, maxV) {
    var html = '<div style="margin:4px 0"><div style="font-size:.78rem;color:var(--fg-dim)">' + title + '</div>';
    html += '<div style="display:flex;align-items:flex-end;gap:3px;height:60px">';
    arr.forEach(function (v) {
      var h = Math.max(3, v / maxV * 56);
      html += '<div title="' + v + '" style="width:14px;height:' + h + 'px;background:' + color +
        ';border:1px solid var(--border);border-radius:3px 3px 0 0"></div>';
    });
    html += "</div></div>";
    return html;
  }

  function render(stage, f) {
    var maxV = Math.max(1, Math.max.apply(null, f.naive.concat([1])));
    var html = bars("naive 每步处理的 token（越来越高 = 越来越慢）", f.naive, "var(--danger)", maxV);
    html += bars("cached 每步处理的 token（几乎恒为 1）", f.cached, "var(--finished)", maxV);
    html += '<div style="font-family:var(--mono);font-size:.82rem;margin-top:6px">累计 naive=' +
      f.nTot + "  cached=" + f.cTot + "</div>";
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    window.Stepper.mount(root, { frames: buildFrames(cfg.promptLen || 6, cfg.nNew || 10), render: render });
  }
  window.RecomputeTimeline = { mount: mount, buildFrames: buildFrames };
})();
