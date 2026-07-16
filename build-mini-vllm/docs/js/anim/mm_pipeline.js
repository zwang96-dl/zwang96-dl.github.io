/* mm_pipeline.js —— 通用的「多模态流水线」单步动画（Lesson 19–30 复用）。
 *
 * 每课传入自己的 stages 列表（[{title, detail}]），逐步高亮当前阶段，展示数据如何一站站
 * 流过多模态管线。纯 DOM + stepper_core，无依赖。
 */
(function () {
  "use strict";
  function render(stage, f, idx) {
    var stages = stage._stages;
    var html = '<div style="display:flex;flex-direction:column;gap:5px;max-width:620px">';
    stages.forEach(function (s, i) {
      var active = i === idx, done = i < idx;
      var border = active ? "3px solid var(--accent)" : "1px solid var(--border)";
      var bg = active ? "color-mix(in srgb, var(--accent) 16%, var(--bg))"
             : done ? "color-mix(in srgb, var(--finished) 10%, var(--bg))" : "var(--bg)";
      html += '<div style="border:' + border + ';background:' + bg +
        ';border-radius:8px;padding:6px 12px">' +
        '<div style="font-weight:600;font-size:.9rem">' + (done ? "✓ " : active ? "▶ " : "") +
        s.title + '</div>' +
        (s.detail ? '<div style="font-size:.78rem;color:var(--fg-dim);font-family:var(--mono)">' +
          s.detail + '</div>' : '') + '</div>';
    });
    stage.innerHTML = html + "</div>";
  }
  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var frames = cfg.stages.map(function (s, i) {
      return { desc: s.title + (s.detail ? "：" + s.detail : ""), code: s.code || "" };
    });
    var stageEl = root.querySelector(".anim-stage");
    if (stageEl) stageEl._stages = cfg.stages;
    window.Stepper.mount(root, { frames: frames, render: render });
  }
  window.MMPipeline = { mount: mount };
})();
