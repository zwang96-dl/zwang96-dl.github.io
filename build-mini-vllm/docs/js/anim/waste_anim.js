/* waste_anim.js —— 通用的「两方案对比」单步动画（Lesson 9/12 复用）。
 *
 * 逐个揭示 items，累计两条方案（a / b）的成本条形，最后给出总量与洞察。
 * mount(sel, {items:[{label,a,b}], aName, bName, insight})。
 */
(function () {
  "use strict";
  function bars(title, vals, color, maxV) {
    var h = '<div style="margin:4px 0"><div style="font-size:.78rem;color:var(--fg-dim)">' + title + "</div>" +
      '<div style="display:flex;align-items:flex-end;gap:4px;height:64px">';
    vals.forEach(function (v) {
      var hh = Math.max(3, v / maxV * 60);
      h += '<div title="' + v + '" style="width:22px;height:' + hh + 'px;background:' + color +
        ';border:1px solid var(--border);border-radius:3px 3px 0 0;position:relative">' +
        '<span style="position:absolute;top:-14px;left:50%;transform:translateX(-50%);' +
        'font-size:.6rem;color:var(--fg-dim)">' + v + '</span></div>';
    });
    return h + "</div></div>";
  }
  function buildFrames(cfg) {
    var items = cfg.items, frames = [];
    var maxV = Math.max.apply(null, items.map(function (it) { return Math.max(it.a, it.b); }));
    for (var i = 1; i <= items.length; i++) {
      var shown = items.slice(0, i);
      frames.push({
        a: shown.map(function (x) { return x.a; }),
        b: shown.map(function (x) { return x.b; }),
        labels: shown.map(function (x) { return x.label; }),
        maxV: maxV,
        desc: "加入 " + items[i - 1].label + "：" + cfg.aName + "=" + items[i - 1].a +
              "，" + cfg.bName + "=" + items[i - 1].b + "。",
        code: cfg.aName + " Σ=" + shown.reduce(function (s, x) { return s + x.a; }, 0) +
              "  " + cfg.bName + " Σ=" + shown.reduce(function (s, x) { return s + x.b; }, 0),
      });
    }
    var ta = items.reduce(function (s, x) { return s + x.a; }, 0);
    var tb = items.reduce(function (s, x) { return s + x.b; }, 0);
    frames.push({
      a: items.map(function (x) { return x.a; }), b: items.map(function (x) { return x.b; }),
      labels: items.map(function (x) { return x.label; }), maxV: maxV,
      desc: cfg.insight + "  合计 " + cfg.aName + "=" + ta + "，" + cfg.bName + "=" + tb + "。",
      code: cfg.aName + "/" + cfg.bName + " = " + ta + " / " + tb,
    });
    return frames;
  }
  function render(stage, f) {
    stage.innerHTML =
      bars((f.aName || "方案A"), f.a, "var(--finished)", f.maxV) +
      bars((f.bName || "方案B"), f.b, "var(--danger)", f.maxV);
  }
  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var frames = buildFrames(cfg);
    frames.forEach(function (fr) { fr.aName = cfg.aName; fr.bName = cfg.bName; });
    window.Stepper.mount(root, { frames: frames, render: render });
  }
  window.WasteAnim = { mount: mount };
})();
