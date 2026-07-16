/* Matrix Multiplication Explorer —— Lesson 2 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   矩阵乘法 C = A @ B 的每个元素是怎么算出来的？为什么形状必须 (m,k)@(k,n)？
 *
 * 它逐个输出格子地演示 row-column rule：高亮 A 的第 i 行与 B 的第 j 列，
 * 展示 out[i][j] = Σ_k A[i][k]·B[k][j] 的乘加过程。镜像 mini_vllm/model/matrix.py 的 matmul()。
 */
(function () {
  "use strict";

  function matmul(A, B) {
    var m = A.length, k = A[0].length, n = B[0].length;
    var C = [];
    for (var i = 0; i < m; i++) { C.push([]); for (var j = 0; j < n; j++) {
      var s = 0; for (var t = 0; t < k; t++) s += A[i][t] * B[t][j]; C[i].push(s);
    } }
    return C;
  }

  function buildFrames(A, B) {
    var m = A.length, k = A[0].length, n = B[0].length;
    var C = matmul(A, B);
    var frames = [{
      i: -1, j: -1, partial: null,
      desc: "A 的形状 (" + m + "," + k + ")，B 的形状 (" + k + "," + n + ")。" +
            "因为 A 的列数(" + k + ") = B 的行数(" + k + ")，两者相容，结果 C 是 (" + m + "," + n + ")。",
      code: "shape(A)=(" + m + "," + k + "), shape(B)=(" + k + "," + n + ") → 相容"
    }];
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        var terms = [];
        for (var t = 0; t < k; t++) terms.push(A[i][t] + "×" + B[t][j]);
        frames.push({
          i: i, j: j, partial: C[i][j],
          desc: "计算 out[" + i + "][" + j + "]：A 的第 " + i + " 行 · B 的第 " + j + " 列 = " +
                terms.join(" + ") + " = " + C[i][j] + "。",
          code: "out[" + i + "][" + j + "] = Σ_k A[" + i + "][k]·B[k][" + j + "] = " + C[i][j]
        });
      }
    }
    frames.push({ i: -1, j: -1, partial: null, done: true,
      desc: "全部 " + (m * n) + " 个格子算完，得到 C。每个格子都是一次「行·列」点积。",
      code: "C = matrix.matmul(A, B)" });
    frames._A = A; frames._B = B; frames._C = C;
    return frames;
  }

  function render(stage, f, idx, total) {
    var A = stage._A, B = stage._B, C = stage._C;
    var S = window.Stepper;
    // 已算出的 C（逐步填充）：显示到当前 frame 为止的格子
    var shown = [];
    for (var r = 0; r < C.length; r++) { shown.push([]); for (var c = 0; c < C[0].length; c++) {
      var order = r * C[0].length + c;
      shown[r].push(order < idx ? C[r][c] : (f.i === r && f.j === c ? C[r][c] : ""));
    } }
    var html =
      '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;">' +
        '<div><div style="font-size:.75rem;color:var(--fg-dim)">A</div>' +
          S.matrixHTML(A, { highlightRow: f.i >= 0 ? f.i : null }) + "</div>" +
        '<div style="font-size:1.2rem">×</div>' +
        '<div><div style="font-size:.75rem;color:var(--fg-dim)">B</div>' +
          S.matrixHTML(B, { highlightCol: f.j >= 0 ? f.j : null }) + "</div>" +
        '<div style="font-size:1.2rem">=</div>' +
        '<div><div style="font-size:.75rem;color:var(--fg-dim)">C</div>' +
          S.matrixHTML(shown, { highlightCell: (f.i >= 0 ? [f.i, f.j] : null) }) + "</div>" +
      "</div>";
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var frames = buildFrames(cfg.A, cfg.B);
    var stage = root.querySelector(".anim-stage");
    if (stage) { stage._A = frames._A; stage._B = frames._B; stage._C = frames._C; }
    window.Stepper.mount(root, { frames: frames, render: render });
  }

  window.MatrixMulExplorer = { mount: mount, buildFrames: buildFrames };
})();
