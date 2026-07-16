/* Attention Stepper —— Lesson 4 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   单头 causal attention 的 5 步 —— QKᵀ → scale → mask → softmax → ×V —— 每一步长什么样？
 *   为什么 causal mask 让第 i 个 query 看不到未来？
 *
 * 镜像 mini_vllm/model/attention_ref.py 的 scaled_dot_product_attention()。
 */
(function () {
  "use strict";
  var NEG = -Infinity;

  function transpose(a) { return a[0].map(function (_, j) { return a.map(function (r) { return r[j]; }); }); }
  function matmul(A, B) {
    var Bt = B, m = A.length, n = B[0].length, k = A[0].length, C = [];
    for (var i = 0; i < m; i++) { C.push([]); for (var j = 0; j < n; j++) {
      var s = 0; for (var t = 0; t < k; t++) s += A[i][t] * B[t][j]; C[i].push(s);
    } } return C;
  }
  function softmaxRow(row) {
    var mx = Math.max.apply(null, row);
    if (mx === NEG) return row.map(function () { return 0; });
    var ex = row.map(function (x) { return x === NEG ? 0 : Math.exp(x - mx); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (e) { return e / s; });
  }

  function buildStages(Q, K, V, causal) {
    var d = Q[0].length;
    var scores = matmul(Q, transpose(K));
    var inv = 1 / Math.sqrt(d);
    var scaled = scores.map(function (r) { return r.map(function (x) { return x * inv; }); });
    var masked = scaled.map(function (r, i) {
      return r.map(function (x, j) { return (causal && j > i) ? NEG : x; });
    });
    var weights = masked.map(softmaxRow);
    var out = matmul(weights, V);
    return { scores: scores, scaled: scaled, masked: masked, weights: weights, out: out, inv: inv, Q: Q, K: K, V: V };
  }

  function buildFrames(Q, K, V, causal) {
    var st = buildStages(Q, K, V, causal);
    return [
      { stage: "inputs", st: st,
        desc: "输入 Q/K/V：每行是一个 token 的向量。Q,K 的列数是 head 维 d=" + Q[0].length + "。",
        code: "Q(Tq,d), K(Tk,d), V(Tk,dv)" },
      { stage: "scores", st: st,
        desc: "第 1 步：scores = Q · Kᵀ。scores[i][j] = 第 i 个 query 与第 j 个 key 的相似度（点积）。",
        code: "scores = matmul(Q, transpose(K))" },
      { stage: "scaled", st: st,
        desc: "第 2 步：除以 √d ≈ " + (1 / st.inv).toFixed(3) + " 做缩放，防止点积随维度变大而过大。",
        code: "scaled = scores / sqrt(d)" },
      { stage: "masked", st: st,
        desc: causal ? "第 3 步：causal mask —— 把 j>i 的位置置为 -∞（自回归不能偷看未来）。"
                     : "第 3 步：非 causal，不加 mask。",
        code: "causal_mask_apply(scaled)  # j>i → -inf" },
      { stage: "weights", st: st,
        desc: "第 4 步：对每一行做 softmax，得到注意力权重（每行非负、和为 1）。-∞ 经 softmax 得 0。",
        code: "weights = softmax(masked, axis=行)" },
      { stage: "out", st: st,
        desc: "第 5 步：out = weights · V —— 用权重对 value 加权求和。第 0 个 query 只看得到 key0。",
        code: "out = matmul(weights, V)" },
    ];
  }

  function render(stage, f) {
    var S = window.Stepper, st = f.st;
    function block(title, m, opts) {
      return '<div style="margin:2px 8px 2px 0"><div style="font-size:.72rem;color:var(--fg-dim)">' +
        title + "</div>" + S.matrixHTML(m, opts || {}) + "</div>";
    }
    var html = '<div style="display:flex;flex-wrap:wrap;align-items:flex-start;">';
    html += block("Q", st.Q) + block("K", st.K) + block("V", st.V);
    html += "</div><div style='display:flex;flex-wrap:wrap;align-items:flex-start;margin-top:6px'>";
    if (f.stage === "scores") html += block("scores = Q·Kᵀ", st.scores);
    if (f.stage === "scaled") html += block("scaled = scores/√d", st.scaled, { prec: 3 });
    if (f.stage === "masked") html += block("masked (未来=-∞)", st.masked, { prec: 3 });
    if (f.stage === "weights") html += block("weights (行和=1)", st.weights, { prec: 3 });
    if (f.stage === "out") {
      html += block("weights", st.weights, { prec: 3 }) + block("out = weights·V", st.out, { prec: 3 });
    }
    html += "</div>";
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var causal = cfg.causal !== false;
    window.Stepper.mount(root, { frames: buildFrames(cfg.Q, cfg.K, cfg.V, causal), render: render });
  }

  window.AttentionStepper = { mount: mount, buildFrames: buildFrames };
})();
