/* blocks_anim.js —— 通用的「KV 物理块」单步动画（Lesson 13/14/16 复用）。
 *
 * 每一帧描述一组物理块的归属状态（owners）与本步新增高亮（newly），由各课在 HTML 内
 * 传入预先算好的 frames。用颜色 + 文字标注 owner（不只靠颜色）。
 */
(function () {
  "use strict";
  function color(idx) {
    if (idx == null) return null;
    var hues = [210, 275, 25, 155, 330, 95, 50, 190];
    return "hsl(" + hues[idx % hues.length] + ", 65%, 55%)";
  }
  function render(stage, f) {
    var owners = f.owners || [];
    var newly = f.newly || [];
    var labels = f.labels || {};   // owner -> 显示文字；owner 为整数索引
    var html = '<div style="display:flex;flex-wrap:wrap;gap:6px;max-width:560px">';
    for (var b = 0; b < owners.length; b++) {
      var o = owners[b];
      var isNew = newly.indexOf(b) >= 0;
      if (o == null || o === "") {
        html += '<div style="width:44px;height:40px;border:2px dashed var(--border);border-radius:6px;' +
          'display:flex;flex-direction:column;align-items:center;justify-content:center;' +
          'font-family:var(--mono);font-size:.62rem;color:var(--fg-dim)">b' + b + '<span>free</span></div>';
      } else {
        var idx = typeof o === "number" ? o : (labels[o] != null ? o : 0);
        var c = color(typeof o === "number" ? o : b);
        var text = labels[o] != null ? labels[o] : o;
        html += '<div style="width:44px;height:40px;border:2px solid ' + c +
          ';background:color-mix(in srgb,' + c + ' 22%, var(--bg));border-radius:6px;' +
          (isNew ? 'box-shadow:0 0 0 3px var(--accent);' : '') +
          'display:flex;flex-direction:column;align-items:center;justify-content:center;' +
          'font-family:var(--mono);font-size:.62rem">b' + b + '<span>' + text + '</span></div>';
      }
    }
    html += "</div>";
    stage.innerHTML = html;
  }
  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    window.Stepper.mount(root, { frames: cfg.frames, render: render });
  }
  window.BlocksAnim = { mount: mount };
})();
