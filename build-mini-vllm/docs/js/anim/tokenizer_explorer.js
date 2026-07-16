/* Tokenizer Explorer —— Lesson 1 的交互式可视化。
 *
 * 这个动画要回答的问题：
 *   一段文本是怎么一步步变成 token id 的？中文/emoji 为什么会变成多个 token？
 *   BOS/PAD 是什么？padding 的 attention mask 从哪来？
 *
 * 它是 mini_vllm/tokenizer.py 中 ByteTokenizer 的镜像：encode 就是「取 UTF-8 字节」，
 * 每个字节即一个 token id（0..255），再加上 BOS(256)/EOS(257)/PAD(258)。
 * 支持实时输入你自己的文本重新演示。
 */
(function () {
  "use strict";
  var BOS = 256, PAD = 258;

  function utf8Bytes(s) {
    // 用浏览器原生 TextEncoder（本地、无网络）取 UTF-8 字节。
    return Array.from(new TextEncoder().encode(s));
  }
  function piece(id) {
    if (id === BOS) return "<bos>";
    if (id === 257) return "<eos>";
    if (id === PAD) return "<pad>";
    var s = new TextDecoder().decode(new Uint8Array([id]));
    return (s && s.charCodeAt(0) >= 32) ? s : "0x" + id.toString(16).toUpperCase();
  }

  function buildFrames(text, padTo) {
    var bytes = utf8Bytes(text);
    var chars = Array.from(text);
    var ids = bytes.slice();               // 每个字节即 id
    var withBos = [BOS].concat(ids);
    var padded = withBos.slice();
    var mask = withBos.map(function () { return 1; });
    while (padded.length < padTo) { padded.push(PAD); mask.push(0); }

    return [
      { view: "chars", data: chars,
        desc: "输入文本共 " + chars.length + " 个字符（注意：字符 ≠ 字节；中文/emoji 一个字符占多个字节）。",
        code: "输入： " + JSON.stringify(text) },
      { view: "bytes", data: bytes,
        desc: "UTF-8 编码：文本变成 " + bytes.length + " 个字节（0..255）。ASCII 一字符=1字节，中文一字符=3字节。",
        code: "text.encode('utf-8')  →  " + bytes.length + " bytes" },
      { view: "ids", data: ids,
        desc: "byte-level：每个字节的数值就是它的 token id。所以此时 token 数 = 字节数 = " + ids.length + "。",
        code: "ByteTokenizer.encode(text, add_bos=False)" },
      { view: "bos", data: withBos,
        desc: "在序列最前面加上 BOS(256) 表示「序列开始」。现在长度 = " + withBos.length + "。",
        code: "ByteTokenizer.encode(text, add_bos=True)" },
      { view: "pad", data: padded, mask: mask,
        desc: "在一个 batch 里，把该序列右侧补 PAD(258) 到长度 " + padTo +
              "，并生成 attention mask（1=真实, 0=PAD）。padding 位置会被 attention 忽略。",
        code: "ByteTokenizer.pad(batch)  →  ids + attention_mask" },
    ];
  }

  function render(stage, f) {
    var items;
    if (f.view === "chars") {
      items = f.data.map(function (c) { return { text: c === " " ? "␠" : c, kind: "normal" }; });
    } else if (f.view === "bytes") {
      items = f.data.map(function (b) { return { text: String(b), sub: piece(b), kind: "normal" }; });
    } else if (f.view === "ids" || f.view === "bos") {
      items = f.data.map(function (id) {
        return { text: String(id), sub: piece(id), kind: id >= 256 ? "special" : "normal" };
      });
    } else { // pad
      items = f.data.map(function (id, i) {
        return { text: String(id), sub: piece(id),
                 kind: id === PAD ? "pad" : (id >= 256 ? "special" : "normal") };
      });
    }
    var html = window.Stepper.chipsHTML(items);
    if (f.view === "pad") {
      html += '<div style="margin-top:8px;font-size:.8rem;color:var(--fg-dim)">attention_mask：</div>';
      html += window.Stepper.chipsHTML(f.mask.map(function (m) {
        return { text: String(m), kind: m ? "normal" : "pad" };
      }));
    }
    stage.innerHTML = html;
  }

  function mount(selector, cfg) {
    var root = document.querySelector(selector);
    if (!root) return;
    var input = root.querySelector('[data-anim-input="text"]');
    var padTo = (cfg && cfg.padTo) || 12;
    var text = (input && input.value) || (cfg && cfg.text) || "你好 🚀";

    var api = window.Stepper.mount(root, { frames: buildFrames(text, padTo), render: render });
    if (input) {
      input.value = text;
      input.addEventListener("input", function () {
        api.rebuild(buildFrames(input.value || "", padTo));
      });
    }
  }

  window.TokenizerExplorer = { mount: mount, buildFrames: buildFrames };
})();
