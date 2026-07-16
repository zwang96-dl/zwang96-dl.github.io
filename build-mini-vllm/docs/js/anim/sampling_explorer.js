/* Sampling Explorer —— Lesson 5 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   logits 怎样一步步变成「下一个 token」？temperature / top-k / top-p 各改变了什么？
 *
 * 镜像 mini_vllm/sampling.py。用一个固定的小 logits 例子逐步演示。
 */
(function () {
  "use strict";
  var LABELS = ["A", "B", "C", "D", "E", "F"];
  var LOGITS = [3.0, 2.2, 1.5, 0.4, -0.5, -1.2];

  function softmax(xs) {
    var m = Math.max.apply(null, xs), ex = xs.map(function (x) { return Math.exp(x - m); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (e) { return e / s; });
  }

  function buildFrames(cfg) {
    var T = cfg.temperature || 0.8, K = cfg.topK || 3, P = cfg.topP || 0.9;
    var probs = softmax(LOGITS);
    var scaled = LOGITS.map(function (x) { return x / T; });
    var pT = softmax(scaled);
    // top-k
    var order = pT.map(function (p, i) { return [p, i]; }).sort(function (a, b) { return b[0] - a[0]; });
    var keepK = new Set(order.slice(0, K).map(function (o) { return o[1]; }));
    var pk = pT.map(function (p, i) { return keepK.has(i) ? p : 0; });
    // top-p on original pT
    var cum = 0, keepP = new Set();
    for (var i = 0; i < order.length; i++) { keepP.add(order[i][1]); cum += order[i][0]; if (cum >= P) break; }
    var pp = pT.map(function (p, i) { return keepP.has(i) ? p : 0; });
    var chosen = order[0][1]; // greedy pick for the final illustration

    return [
      { view: "logits", data: LOGITS, desc: "模型输出 logits（原始分数，未归一）。分数最高的是 " + LABELS[chosen] + "。",
        code: "logits = model.forward(...)[-1]" },
      { view: "probs", data: probs, desc: "greedy 直接取分数最高的（= argmax）。temperature=0 时就到此为止。",
        code: "argmax(logits)" },
      { view: "scaled", data: pT, desc: "temperature=" + T + "：logits/T 后再 softmax。T>1 更平（随机），T<1 更尖（保守）。",
        code: "softmax(logits / T)" },
      { view: "topk", data: pk, desc: "top-k=" + K + "：只在概率最高的 " + K + " 个里采样，其余置 0。",
        code: "top_k_filter(probs, k)" },
      { view: "topp", data: pp, desc: "top-p=" + P + "：保留累计概率达 " + P + " 的最小集合（nucleus）。",
        code: "top_p_filter(probs, p)" },
      { view: "sample", data: pp, chosen: chosen, desc: "在过滤后的分布里按概率采样，得到下一个 token。",
        code: "Sampler(params)(logits)" },
    ];
  }

  function render(stage, f) {
    var data = f.data, maxV = Math.max.apply(null, data.map(Math.abs)) || 1;
    var html = '<div style="display:flex;align-items:flex-end;gap:10px;height:130px;padding:8px">';
    for (var i = 0; i < data.length; i++) {
      var v = data[i];
      var h = Math.max(2, Math.abs(v) / maxV * 100);
      var zero = (f.view !== "logits" && v === 0);
      var chosen = (f.view === "sample" && f.chosen === i);
      var bg = chosen ? "var(--accent)" : zero ? "var(--border)" : "var(--running)";
      html += '<div style="display:flex;flex-direction:column;align-items:center;gap:3px">' +
        '<div style="font-family:var(--mono);font-size:.7rem;color:var(--fg-dim)">' +
          (f.view === "logits" ? v.toFixed(1) : (v === 0 ? "·" : v.toFixed(2))) + '</div>' +
        '<div style="width:30px;height:' + h + 'px;background:' + bg +
          ';border:' + (chosen ? "2px solid var(--accent-fg)" : "1px solid var(--border)") + ';border-radius:4px 4px 0 0"></div>' +
        '<div style="font-family:var(--mono);font-weight:700">' + LABELS[i] +
          (chosen ? " ✓" : "") + '</div></div>';
    }
    html += "</div>";
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    window.Stepper.mount(root, { frames: buildFrames(cfg || {}), render: render });
  }
  window.SamplingExplorer = { mount: mount, buildFrames: buildFrames };
})();
