/* Build Your Own Mini-vLLM —— 共享脚本（纯原生 JS，无任何依赖）。
   职责：主题切换、TOC 高亮、阅读进度（localStorage）。
   注意：网页进度只用于「我读到哪了」，与 course.py 的进度是两套独立记录，
   网页绝不会修改你的源代码。 */
(function () {
  "use strict";

  // ---------- 主题 ----------
  var THEME_KEY = "mini-vllm-theme";
  function applyTheme(t) {
    if (t === "light" || t === "dark") {
      document.documentElement.setAttribute("data-theme", t);
    } else {
      document.documentElement.removeAttribute("data-theme"); // 跟随系统
    }
  }
  function currentTheme() { return localStorage.getItem(THEME_KEY) || "system"; }
  function cycleTheme() {
    var order = ["system", "light", "dark"];
    var next = order[(order.indexOf(currentTheme()) + 1) % order.length];
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
    updateThemeButton();
  }
  function updateThemeButton() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var t = currentTheme();
    var label = t === "system" ? "主题：跟随系统" : t === "light" ? "主题：浅色" : "主题：深色";
    btn.textContent = label;
    btn.setAttribute("aria-label", label);
  }
  applyTheme(currentTheme());

  // ---------- 进度（localStorage） ----------
  var PROG_KEY = "mini-vllm-progress";
  function getProgress() {
    try { return JSON.parse(localStorage.getItem(PROG_KEY)) || {}; }
    catch (e) { return {}; }
  }
  function markRead(lesson) {
    var p = getProgress();
    p[lesson] = true;
    localStorage.setItem(PROG_KEY, JSON.stringify(p));
    renderProgress();
  }
  function renderProgress() {
    var p = getProgress();
    var total = parseInt(document.body.getAttribute("data-total-lessons") || "31", 10);
    var count = Object.keys(p).filter(function (k) { return p[k]; }).length;
    var bar = document.getElementById("progress-bar");
    var txt = document.getElementById("progress-text");
    if (bar) bar.style.width = Math.min(100, (count / total) * 100) + "%";
    if (txt) txt.textContent = count + " / " + total + " 课已标记学习";
    // 首页课程项打勾
    document.querySelectorAll("[data-lesson-mark]").forEach(function (el) {
      var n = el.getAttribute("data-lesson-mark");
      el.textContent = p[n] ? "✓ 已学" : "";
    });
  }

  // ---------- TOC 高亮 ----------
  function setupTOC() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#']"));
    if (!links.length) return;
    var map = {};
    links.forEach(function (a) {
      var id = a.getAttribute("href").slice(1);
      var sec = document.getElementById(id);
      if (sec) map[id] = a;
    });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (l) { l.classList.remove("active"); });
          var a = map[e.target.id];
          if (a) a.classList.add("active");
        }
      });
    }, { rootMargin: "-10% 0px -80% 0px" });
    Object.keys(map).forEach(function (id) { obs.observe(document.getElementById(id)); });
  }

  // ---------- 初始化 ----------
  document.addEventListener("DOMContentLoaded", function () {
    updateThemeButton();
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", cycleTheme);
    setupTOC();
    renderProgress();

    // 「标记本课已学」按钮
    document.querySelectorAll("[data-mark-lesson]").forEach(function (b) {
      b.addEventListener("click", function () { markRead(b.getAttribute("data-mark-lesson")); });
    });

    // 首页搜索
    var search = document.getElementById("lesson-search");
    if (search) {
      search.addEventListener("input", function () {
        var q = search.value.trim().toLowerCase();
        document.querySelectorAll("[data-search]").forEach(function (item) {
          var hay = item.getAttribute("data-search").toLowerCase();
          item.style.display = hay.indexOf(q) >= 0 ? "" : "none";
        });
      });
    }
  });

  // 暴露给页面（可选）
  window.MiniVLLM = { markRead: markRead, getProgress: getProgress };
})();
