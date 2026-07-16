/* stepper_core.js —— 可复用的「单步动画」内核（纯原生 JS，无依赖）。
 *
 * 处理所有动画通用的部分：Play/Pause/Next/Prev/Reset、速度、键盘、进度、
 * 复用页面里已有的静态骨架（含 data-anim-control 按钮）。各课的动画只需提供
 * 一个 frames 数组和一个 render(stage, frame) 函数即可。
 *
 * 还提供两个渲染小工具：matrixHTML（把矩阵画成带高亮的表格）、chipsHTML
 * （把一串值画成一排 chip）。区分靠「文字 + 边框/背景」，不只靠颜色。
 */
(function () {
  "use strict";

  function esc(s) { return String(s).replace(/[&<>]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }

  function fmt(x, prec) {
    if (x === Infinity) return "∞";
    if (x === -Infinity) return "-∞";
    if (typeof x === "number" && !Number.isInteger(x)) return x.toFixed(prec == null ? 3 : prec);
    return String(x);
  }

  // 把矩阵渲染成 HTML 表格，可高亮某行/某列/某格。
  function matrixHTML(m, opts) {
    opts = opts || {};
    var hr = opts.highlightRow, hc = opts.highlightCol, hcell = opts.highlightCell;
    var prec = opts.prec;
    var out = '<table class="mtx" style="border-collapse:collapse;display:inline-table;margin:2px 6px;">';
    for (var i = 0; i < m.length; i++) {
      out += "<tr>";
      for (var j = 0; j < m[i].length; j++) {
        var hi = (hr === i) || (hc === j) || (hcell && hcell[0] === i && hcell[1] === j);
        var strong = hcell && hcell[0] === i && hcell[1] === j;
        var bg = strong ? "var(--accent)" : hi ? "color-mix(in srgb, var(--accent) 22%, var(--bg))" : "var(--bg)";
        var fg = strong ? "var(--accent-fg)" : "var(--fg)";
        out += '<td style="border:1px solid var(--border);padding:4px 8px;text-align:right;' +
          "font-family:var(--mono);font-size:.82rem;background:" + bg + ";color:" + fg + ';">' +
          esc(fmt(m[i][j], prec)) + "</td>";
      }
      out += "</tr>";
    }
    return out + "</table>";
  }

  // 把一串值渲染成 chip（可给每个 chip 一个 label 与样式类）。
  function chipsHTML(items) {
    // items: [{text, sub, kind}]  kind: normal|special|pad
    return '<div style="display:flex;flex-wrap:wrap;gap:6px;">' + items.map(function (it) {
      var border = it.kind === "special" ? "3px double var(--prefill)"
        : it.kind === "pad" ? "2px dashed var(--waiting)" : "2px solid var(--running)";
      return '<div style="min-width:34px;text-align:center;border:' + border +
        ';border-radius:6px;padding:4px 6px;font-family:var(--mono);font-size:.8rem;background:var(--bg);">' +
        esc(it.text) + (it.sub != null ? '<div style="font-size:.68rem;color:var(--fg-dim)">' + esc(it.sub) + "</div>" : "") +
        "</div>";
    }).join("") + "</div>";
  }

  function mount(root, opts) {
    if (typeof root === "string") root = document.querySelector(root);
    if (!root) return null;
    var frames = opts.frames || [];
    var stage = root.querySelector(".anim-stage");
    var elNo = root.querySelector(".step-no");
    var elText = root.querySelector(".step-text");
    var elCode = root.querySelector(".step-code");
    var elProg = root.querySelector(".rl-progress");
    var btnPlay = root.querySelector('[data-anim-control="play"]');
    var btnPause = root.querySelector('[data-anim-control="pause"]');
    var idx = 0, timer = null, speed = 1, playing = false;

    root.setAttribute("tabindex", "0");
    root.setAttribute("role", "group");

    function render() {
      var f = frames[idx];
      if (opts.render && stage) opts.render(stage, f, idx, frames.length);
      if (elNo) elNo.textContent = "步骤 " + idx + " / " + (frames.length - 1);
      if (elText) elText.textContent = f && f.desc ? f.desc : "";
      if (elCode) elCode.textContent = f && f.code ? "对应代码： " + f.code : "";
      if (elProg) elProg.textContent = idx + "/" + (frames.length - 1);
    }
    function go(n) { idx = Math.max(0, Math.min(frames.length - 1, n)); render(); if (idx >= frames.length - 1) stop(); }
    function next() { if (idx < frames.length - 1) go(idx + 1); else stop(); }
    function prev() { go(idx - 1); }
    function reset() { stop(); go(0); }
    function tick() { if (idx >= frames.length - 1) { stop(); return; } next(); timer = setTimeout(tick, 1200 / speed); }
    function play() { if (idx >= frames.length - 1) idx = 0; playing = true; if (btnPlay) btnPlay.hidden = true; if (btnPause) btnPause.hidden = false; timer = setTimeout(tick, 1200 / speed); }
    function stop() { playing = false; if (btnPlay) btnPlay.hidden = false; if (btnPause) btnPause.hidden = true; if (timer) { clearTimeout(timer); timer = null; } }

    function bind(sel, fn) { var el = root.querySelector(sel); if (el) el.addEventListener("click", fn); }
    bind('[data-anim-control="next"]', function () { stop(); next(); });
    bind('[data-anim-control="prev"]', function () { stop(); prev(); });
    bind('[data-anim-control="reset"]', reset);
    if (btnPlay) btnPlay.addEventListener("click", play);
    if (btnPause) btnPause.addEventListener("click", stop);
    var spd = root.querySelector('[data-anim-control="speed"]');
    if (spd) spd.addEventListener("input", function (e) { speed = parseFloat(e.target.value) || 1; });

    root.addEventListener("keydown", function (e) {
      if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stop(); next(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); stop(); prev(); }
      else if (e.key === "r" || e.key === "R") { reset(); }
    });

    render();
    return {
      render: render, go: go,
      rebuild: function (newFrames) { stop(); frames = newFrames; idx = 0; render(); }
    };
  }

  window.Stepper = { mount: mount, matrixHTML: matrixHTML, chipsHTML: chipsHTML, fmt: fmt };
})();
