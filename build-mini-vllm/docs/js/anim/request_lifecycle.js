/* Request Lifecycle 动画 —— Lesson 0 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   一个请求从到达到结束，经历了哪些状态？prefill 和 decode 有何不同？
 *   KV block 是何时分配、何时释放的？
 *
 * 它是 mini_vllm/simulator/text_pipeline.py 中 LifecycleSimulator 的**忠实镜像**：
 *   - 相同的准入规则（受 max_num_seqs 限制、按 arrival 到达）
 *   - 相同的调度（token 预算内先 prefill 后 decode）
 *   - 相同的 KV block 分配（ceil(上下文长度 / block_size)，结束后释放）
 * 因此你在网页里看到的每一步，都能在 Python 侧用 `--trace` 或 `course.py inspect` 复现。
 *
 * 纯原生 JS + DOM，无任何依赖，可离线运行。
 */
(function () {
  "use strict";

  function ceilDiv(a, b) { return Math.max(1, Math.ceil(a / b)); }

  // 按请求下标给一个稳定、可区分的色相（同时会用文字标注 owner，不只靠颜色）。
  function ownerColor(idx) {
    var hues = [210, 275, 25, 155, 330, 95];
    return "hsl(" + hues[idx % hues.length] + ", 65%, 55%)";
  }

  /* ---- 生命周期求解：产出逐步快照 frames ---- */
  function buildFrames(cfg) {
    var bs = cfg.blockSize, nb = cfg.numBlocks;
    // 用 UTF-8 字节数 + 1(BOS) 计算 prompt token 数，与 Python 的 byte-level tokenizer 一致
    // （中文一字 3 字节 = 3 个 token），保证动画是 `run 0 --trace` 的忠实镜像。
    var byteLen = function (s) {
      return (typeof TextEncoder !== "undefined") ? new TextEncoder().encode(s).length : s.length;
    };
    var reqs = cfg.requests.map(function (r, i) {
      return {
        id: r.id, idx: i,
        promptLen: byteLen(r.prompt) + 1, // +1 = BOS
        prompt: r.prompt,
        maxNew: r.maxNew, arrival: r.arrival || 0,
        gen: 0, state: "WAITING", phase: null, blocks: [], firstTokenIter: null
      };
    });
    var byId = {}; reqs.forEach(function (r) { byId[r.id] = r; });

    var free = []; for (var i = 0; i < nb; i++) free.push(i);
    var owner = new Array(nb).fill(null); // blockId -> reqId

    function snapshot(desc, codeMap, extra) {
      var states = {};
      reqs.forEach(function (r) {
        states[r.id] = {
          state: r.state, phase: r.phase,
          ctx: r.promptLen + r.gen, promptLen: r.promptLen, gen: r.gen,
          blocks: r.blocks.slice()
        };
      });
      return Object.assign({
        desc: desc, codeMap: codeMap || "",
        states: states, owner: owner.slice(),
        inUse: nb - free.length, freeCount: free.length
      }, extra || {});
    }

    var frames = [snapshot(
      "初始状态：两个请求都在 WAITING 队列里，还没有被调度，也没有分配任何 KV block。",
      "mini_vllm/simulator/text_pipeline.py · SimRequest(state=WAITING)",
      { iter: 0, scheduled: [], schedTokens: 0, newAllocs: [], finished: [] }
    )];

    var pending = reqs.slice().sort(function (a, b) {
      return a.arrival - b.arrival || (a.id < b.id ? -1 : 1);
    });
    var running = [];
    var iter = 0, guard = 0;

    while (pending.length || running.length) {
      if (++guard > 1000) break;
      iter++;

      // 1) 准入
      var admitted = [];
      while (pending.length && running.length < cfg.maxNumSeqs && pending[0].arrival < iter) {
        var r = pending.shift();
        r.state = "RUNNING"; r.phase = "prefill";
        running.push(r); admitted.push(r.id);
      }
      if (running.length === 0) {
        frames.push(snapshot(
          "迭代 " + iter + "：还没有请求到达（arrival 在未来），引擎空转一步推进时间。",
          "LifecycleSimulator.run() · idle-iteration",
          { iter: iter, scheduled: [], schedTokens: 0, newAllocs: [], finished: [] }));
        continue;
      }

      // 2) 调度（token 预算内，先 prefill 后 decode）
      var budget = cfg.tokenBudget, scheduled = [], schedTokens = 0;
      running.forEach(function (r) {
        var cost = r.phase === "prefill" ? r.promptLen : 1;
        if (cost <= budget) { budget -= cost; schedTokens += cost; scheduled.push(r); }
      });

      // 3) 每个被调度请求：分配 block + 前向一步
      var newAllocs = [];
      scheduled.forEach(function (r) {
        var need = ceilDiv(r.promptLen + r.gen, bs);
        while (r.blocks.length < need) {
          if (!free.length) throw new Error("KV blocks 耗尽（模拟 OOM）");
          var phys = free.shift();
          owner[phys] = r.id;
          newAllocs.push({ req: r.id, logical: r.blocks.length, phys: phys });
          r.blocks.push(phys);
        }
        if (r.phase === "prefill") { r.gen++; r.firstTokenIter = iter; r.phase = "decode"; }
        else { r.gen++; }
      });

      // 4) 收尾：写满的请求 FINISHED 并释放 block（演示无泄漏）
      var finished = [];
      running = running.filter(function (r) {
        if (r.gen >= r.maxNew) {
          r.state = "FINISHED";
          r.blocks.forEach(function (p) { owner[p] = null; free.push(p); });
          r.blocks = []; finished.push(r.id);
          return false;
        }
        return true;
      });
      free.sort(function (a, b) { return a - b; });

      // 组织本步的说明与代码映射
      var parts = [];
      if (admitted.length) parts.push("准入 " + admitted.join("、") + "（WAITING → RUNNING）");
      var phaseDesc = scheduled.map(function (r) {
        var st = frames.length; // unused
        return r.id + "(" + (r.firstTokenIter === iter ? "PREFILL：一次吃掉整段 prompt " + r.promptLen + " tokens" : "DECODE：只处理 1 个 token") + ")";
      });
      if (scheduled.length) parts.push("调度 " + phaseDesc.join("；") + "，本迭代共 " + schedTokens + " tokens");
      if (finished.length) parts.push("结束 " + finished.join("、") + "（FINISHED，归还其 KV block —— 无泄漏）");

      var code = "";
      if (newAllocs.length) {
        code = newAllocs.map(function (a) {
          return "为 " + a.req + " 分配 logical block " + a.logical + " → physical block " + a.phys;
        }).join("；") + "  ⟶  BlockAllocator.allocate()";
      } else if (finished.length) {
        code = "释放 " + finished.join("、") + " 的 physical block  ⟶  BlockAllocator.free()";
      } else {
        code = "ModelRunner.forward()  ⟶  input_ids.shape = [1, " +
          (scheduled.length && scheduled[0].firstTokenIter === iter ? scheduled[0].promptLen : 1) + "]";
      }

      frames.push(snapshot("迭代 " + iter + "：" + parts.join("；") + "。",
        code, { iter: iter, scheduled: scheduled.map(function (r) { return r.id; }),
                schedTokens: schedTokens, newAllocs: newAllocs, finished: finished }));
    }

    frames.push(snapshot(
      "全部请求 FINISHED。注意：正在使用的 KV block 已归零——结束即释放，没有泄漏。这正是 Lesson 13 BlockAllocator 要保证的性质。",
      "assert pool.in_use == 0  （无 KV block 泄漏）",
      { iter: iter + 1, scheduled: [], schedTokens: 0, newAllocs: [], finished: [] }));

    return { frames: frames, reqs: reqs, cfg: cfg };
  }

  /* ---- 渲染 ---- */
  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var model = buildFrames(cfg);
    var frames = model.frames;
    var idx = 0, playing = false, timer = null, speed = 1;

    root.setAttribute("tabindex", "0");
    root.setAttribute("role", "group");
    root.setAttribute("aria-label", "请求生命周期动画，可用空格播放/暂停，左右方向键单步，R 重置");

    // DOM 结构：若页面已提供静态骨架（含 data-anim-control 按钮），直接复用；
    // 否则在此注入（便于其他页面无骨架时也能用）。静态骨架保证「无 JS 也能看到控件」
    // 与「测试可在静态 HTML 中校验控件存在」。
    if (!root.querySelector('[data-anim-control="play"]'))
    root.innerHTML =
      '<div class="anim-head"><div class="anim-q">这个动画要回答的问题：' +
      '<small>一个请求经历哪些状态？prefill 与 decode 有何不同？KV block 何时分配/释放？</small></div></div>' +
      '<div class="anim-stage">' +
        '<div class="rl-scheduler" style="font-family:var(--mono);font-size:.85rem;margin-bottom:10px;"></div>' +
        '<div class="rl-reqs" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;"></div>' +
        '<div style="font-size:.8rem;color:var(--fg-dim);margin:6px 0 4px;">KV Block 池（物理 block，' + cfg.numBlocks + ' 个，block_size=' + cfg.blockSize + '）：</div>' +
        '<div class="rl-blocks" style="display:grid;grid-template-columns:repeat(' + Math.min(cfg.numBlocks, 8) + ',1fr);gap:6px;max-width:520px;"></div>' +
      '</div>' +
      '<div class="legend">' +
        '<span class="item"><span class="swatch waiting"></span>WAITING（虚线）</span>' +
        '<span class="item"><span class="swatch running"></span>RUNNING（实线）</span>' +
        '<span class="item"><span class="swatch prefill"></span>PREFILL（双线）</span>' +
        '<span class="item"><span class="swatch decode"></span>DECODE（点线）</span>' +
        '<span class="item"><span class="swatch finished"></span>FINISHED（圆形）</span>' +
      '</div>' +
      '<div class="anim-step-desc"><span class="step-no"></span><span class="step-text"></span>' +
        '<div class="step-code" style="font-family:var(--mono);font-size:.82rem;color:var(--fg-dim);margin-top:4px;"></div></div>' +
      '<div class="anim-controls">' +
        '<button class="btn" data-anim-control="prev" aria-label="上一步">⏮ 上一步</button>' +
        '<button class="btn btn-accent" data-anim-control="play" aria-label="播放">▶ 播放</button>' +
        '<button class="btn" data-anim-control="pause" aria-label="暂停" hidden>⏸ 暂停</button>' +
        '<button class="btn" data-anim-control="next" aria-label="下一步">下一步 ⏭</button>' +
        '<button class="btn" data-anim-control="reset" aria-label="重置">↺ 重置</button>' +
        '<span class="spacer"></span>' +
        '<label>速度 <input type="range" data-anim-control="speed" min="0.5" max="3" step="0.5" value="1"></label>' +
        '<span class="rl-progress" style="font-family:var(--mono);font-size:.8rem;color:var(--fg-dim);"></span>' +
      '</div>';

    // 若页面用的是通用骨架（只有空的 .anim-stage 与控件），补齐生命周期动画专用的舞台结构。
    var stage0 = root.querySelector(".anim-stage");
    if (stage0 && !root.querySelector(".rl-scheduler")) {
      stage0.innerHTML =
        '<div class="rl-scheduler" style="font-family:var(--mono);font-size:.85rem;margin-bottom:10px;">（加载中…）</div>' +
        '<div class="rl-reqs" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;"></div>' +
        '<div style="font-size:.8rem;color:var(--fg-dim);margin:6px 0 4px;">KV Block 池（物理 block，' +
          cfg.numBlocks + ' 个，block_size=' + cfg.blockSize + '）：</div>' +
        '<div class="rl-blocks" style="display:grid;grid-template-columns:repeat(' +
          Math.min(cfg.numBlocks, 8) + ',1fr);gap:6px;max-width:520px;"></div>';
    }

    var elSched = root.querySelector(".rl-scheduler");
    var elReqs = root.querySelector(".rl-reqs");
    var elBlocks = root.querySelector(".rl-blocks");
    var elStepNo = root.querySelector(".step-no");
    var elStepText = root.querySelector(".step-text");
    var elStepCode = root.querySelector(".step-code");
    var elProgress = root.querySelector(".rl-progress");
    var btnPlay = root.querySelector('[data-anim-control="play"]');
    var btnPause = root.querySelector('[data-anim-control="pause"]');

    function stateStyle(s) {
      // 用边框样式（形状）+ 颜色共同表达状态，满足「不只靠颜色」。
      if (s.state === "WAITING") return { border: "2px dashed var(--waiting)", radius: "8px" };
      if (s.state === "FINISHED") return { border: "2px solid var(--finished)", radius: "50%" };
      if (s.phase === "prefill") return { border: "3px double var(--prefill)", radius: "8px" };
      if (s.phase === "decode") return { border: "2px dotted var(--decode)", radius: "8px" };
      return { border: "2px solid var(--running)", radius: "8px" };
    }

    function render() {
      var f = frames[idx];

      // scheduler 行
      if (f.scheduled.length) {
        elSched.innerHTML = "Scheduler · 迭代 " + f.iter +
          "：scheduled = [" + f.scheduled.join(", ") + "]，scheduled_tokens = " + f.schedTokens;
      } else {
        elSched.innerHTML = "Scheduler · 迭代 " + f.iter + "：（本步无调度）";
      }

      // 请求卡片
      elReqs.innerHTML = "";
      model.reqs.forEach(function (r) {
        var s = f.states[r.id];
        var sty = stateStyle(s);
        var card = document.createElement("div");
        card.style.cssText = "min-width:150px;padding:10px 12px;background:var(--bg);" +
          "border:" + sty.border + ";border-radius:" + sty.radius + ";";
        var phaseTxt = s.state === "RUNNING" ? " · " + (s.phase || "").toUpperCase() : "";
        card.innerHTML =
          '<div style="font-weight:700;font-family:var(--mono)">' + r.id + '</div>' +
          '<div style="font-size:.8rem;color:var(--fg-dim)">' + s.state + phaseTxt + '</div>' +
          '<div style="font-size:.8rem;margin-top:4px">ctx = ' + s.ctx +
            ' <span style="color:var(--fg-dim)">(prompt ' + s.promptLen + ' + gen ' + s.gen + ')</span></div>' +
          '<div style="font-size:.8rem">blocks = [' + s.blocks.join(", ") + ']</div>';
        elReqs.appendChild(card);
      });

      // KV block 网格
      elBlocks.innerHTML = "";
      for (var b = 0; b < cfg.numBlocks; b++) {
        var own = f.owner[b];
        var isNew = f.newAllocs.some(function (a) { return a.phys === b; });
        var cell = document.createElement("div");
        var base = "height:46px;border-radius:6px;display:flex;flex-direction:column;" +
          "align-items:center;justify-content:center;font-family:var(--mono);font-size:.72rem;";
        if (own == null) {
          cell.style.cssText = base + "border:2px dashed var(--border);color:var(--fg-dim);";
          cell.innerHTML = "b" + b + "<span>free</span>";
        } else {
          var idxOwner = model.reqs.findIndex(function (r) { return r.id === own; });
          cell.style.cssText = base + "border:2px solid " + ownerColor(idxOwner) +
            ";background:color-mix(in srgb," + ownerColor(idxOwner) + " 22%, var(--bg));" +
            (isNew ? "box-shadow:0 0 0 3px var(--accent);" : "");
          cell.innerHTML = "b" + b + "<span>" + own + "</span>";
        }
        elBlocks.appendChild(cell);
      }

      // 说明 + 代码映射
      elStepNo.textContent = "步骤 " + idx + " / " + (frames.length - 1);
      elStepText.textContent = f.desc;
      elStepCode.textContent = f.codeMap ? "对应代码： " + f.codeMap : "";
      elProgress.textContent = idx + "/" + (frames.length - 1);
    }

    function go(n) {
      idx = Math.max(0, Math.min(frames.length - 1, n));
      render();
      if (idx >= frames.length - 1) stop();
    }
    function next() { if (idx < frames.length - 1) go(idx + 1); else stop(); }
    function prev() { go(idx - 1); }
    function reset() { stop(); go(0); }

    function tick() {
      if (idx >= frames.length - 1) { stop(); return; }
      next();
      timer = setTimeout(tick, 1200 / speed);
    }
    function play() {
      if (idx >= frames.length - 1) idx = 0;
      playing = true; btnPlay.hidden = true; btnPause.hidden = false;
      timer = setTimeout(tick, 1200 / speed);
    }
    function stop() {
      playing = false; btnPlay.hidden = false; btnPause.hidden = true;
      if (timer) { clearTimeout(timer); timer = null; }
    }

    root.querySelector('[data-anim-control="next"]').addEventListener("click", function () { stop(); next(); });
    root.querySelector('[data-anim-control="prev"]').addEventListener("click", function () { stop(); prev(); });
    btnPlay.addEventListener("click", play);
    btnPause.addEventListener("click", stop);
    root.querySelector('[data-anim-control="reset"]').addEventListener("click", reset);
    root.querySelector('[data-anim-control="speed"]').addEventListener("input", function (e) {
      speed = parseFloat(e.target.value) || 1;
    });

    root.addEventListener("keydown", function (e) {
      if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stop(); next(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); stop(); prev(); }
      else if (e.key === "r" || e.key === "R") { reset(); }
    });

    render();
  }

  window.RequestLifecycleAnim = { mount: mount, buildFrames: buildFrames };
})();
