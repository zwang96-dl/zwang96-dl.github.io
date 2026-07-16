/* Transformer Pipeline —— Lesson 3 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   一次前向里，数据依次流过哪些子层？每一步的 shape 是什么？残差在哪里相加？
 *
 * 镜像 mini_vllm/model/transformer.py 的 TinyTextModel.forward()。
 */
(function () {
  "use strict";

  function buildFrames(cfg) {
    var H = cfg.hidden, L = cfg.layers, seq = cfg.seq, V = cfg.vocab;
    var f = [];
    function add(name, shape, desc, code) { f.push({ name: name, shape: shape, desc: desc, code: code }); }
    add("input_ids", "[" + seq + "]",
        "输入是 " + seq + " 个 token id。", "input_ids");
    add("Embedding 查表", "(" + seq + ", " + H + ")",
        "每个 token id 查 embedding 表，得到一个 hidden 维向量。", "h = embed[input_ids]");
    for (var l = 0; l < L; l++) {
      add("Layer " + l + " · RMSNorm", "(" + seq + ", " + H + ")",
          "层 " + l + "：先 RMSNorm 稳定尺度。", "rms_norm(h, ln1)");
      add("Layer " + l + " · Q/K/V 投影", "Q(" + seq + "," + H + ") K/V(" + seq + "," + (cfg.kvHeads * cfg.headDim) + ")",
          "投影出 Q、K、V。GQA：K/V 的 head 更少（省 KV 内存）。", "matmul(hn, Wq/Wk/Wv)");
      add("Layer " + l + " · RoPE", "(" + seq + ", " + H + ")",
          "给 Q、K 注入位置信息（旋转）。", "apply_rope_heads(q, k)");
      add("Layer " + l + " · Causal Attention", "(" + seq + ", " + H + ")",
          "每个位置对「当前及之前」的 K/V 做加权求和。", "sdpa_positions(...)  →  Wo");
      add("Layer " + l + " · 残差①", "(" + seq + ", " + H + ")",
          "h = h + attention 输出（残差连接，保梯度/信息）。", "h = h + o");
      add("Layer " + l + " · RMSNorm→SwiGLU", "(" + seq + ", " + H + ")",
          "再归一化，过 SwiGLU 前馈网络。", "swiglu(rms_norm(h, ln2))");
      add("Layer " + l + " · 残差②", "(" + seq + ", " + H + ")",
          "h = h + MLP 输出。", "h = h + mlp");
    }
    add("最终 RMSNorm", "(" + seq + ", " + H + ")", "所有层之后做一次最终归一化。", "rms_norm(h, final_ln)");
    add("LM head", "(" + seq + ", " + V + ")",
        "投影到词表维度，得到 logits。最后一行用于预测下一个 token。", "logits = matmul(hf, lm_head)");
    // 附上 desc 里的 shape 说明
    f.forEach(function (fr) { fr.desc = fr.desc + "  shape = " + fr.shape; });
    return f;
  }

  function render(stage, f, idx, total) {
    var frames = stage._frames;
    var html = '<div style="display:flex;flex-direction:column;gap:4px;max-width:520px">';
    frames.forEach(function (fr, i) {
      var active = i === idx, done = i < idx;
      var border = active ? "3px solid var(--accent)" : "1px solid var(--border)";
      var bg = active ? "color-mix(in srgb, var(--accent) 18%, var(--bg))"
             : done ? "color-mix(in srgb, var(--finished) 10%, var(--bg))" : "var(--bg)";
      html += '<div style="display:flex;justify-content:space-between;gap:8px;border:' + border +
        ';background:' + bg + ';border-radius:8px;padding:5px 10px;font-size:.85rem">' +
        '<span>' + (done ? "✓ " : active ? "▶ " : "") + fr.name + '</span>' +
        '<span style="font-family:var(--mono);color:var(--fg-dim)">' + fr.shape + '</span></div>';
    });
    html += "</div>";
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var frames = buildFrames(cfg);
    var stage = root.querySelector(".anim-stage");
    if (stage) stage._frames = frames;
    window.Stepper.mount(root, { frames: frames, render: render });
  }
  window.TransformerPipeline = { mount: mount, buildFrames: buildFrames };
})();
