"""一次性机械校验：
1) 每个课程页面里所有 <math>…</math> 块是否是良构 XML（能被解析）。
2) 是否还有 <pre> 显示块残留「显示型数学公式」符号（Σ ∑ √ ⌈ ⌉ 上标转置等）——
   这类才是应该转成 MathML 的；命令/代码/流程图不算。
3) 统计每课 <math> 块数。
用法：python3 scripts/check_mathml.py
"""
import re
import sys
import pathlib
import xml.etree.ElementTree as ET

LESSONS = pathlib.Path("docs/lessons")

# 显示型公式的强信号符号（命令/代码里几乎不会成块出现）
MATH_SIGNS = ["∑", "Σ", "√", "⌈", "⌉", "ᵀ", "≤", "≥", "≠", "√"]
# 上标/下标 Unicode 常见字符
SUP_SUB = "⁰¹²³⁴⁵⁶⁷⁸⁹ᵀₖₙᵢⱼₐ"

math_block_re = re.compile(r"<math\b[^>]*>.*?</math>", re.DOTALL)
pre_block_re = re.compile(r"<pre\b([^>]*)>(.*?)</pre>", re.DOTALL)


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def main() -> int:
    total_math = 0
    malformed = []
    suspicious_pre = []
    per_lesson = {}

    for path in sorted(LESSONS.glob("lesson_*.html")):
        html = path.read_text(encoding="utf-8")
        maths = math_block_re.findall(html)
        per_lesson[path.name] = len(maths)
        total_math += len(maths)

        for i, m in enumerate(maths):
            try:
                # MathML 无命名空间前缀，可直接当 XML 解析；&nbsp; 等实体需替换
                probe = m.replace("&nbsp;", " ").replace("&times;", "×").replace("&sdot;", "·")
                ET.fromstring(probe)
            except ET.ParseError as e:
                malformed.append((path.name, i, str(e), m[:120]))

        # 检查 <pre> 里是否还有显示型公式
        for attrs, body in pre_block_re.findall(html):
            if "cmd" in attrs:
                continue  # 命令块，跳过
            text = strip_tags(body)
            # 代码信号：有这些就当作代码，不算公式
            code_signals = ["def ", "import ", "self.", "return ", "course.py",
                            "= [", "].append", "for ", "print(", "python3"]
            if any(cs in text for cs in code_signals):
                continue
            has_sign = any(s in text for s in MATH_SIGNS)
            has_supsub = any(c in text for c in SUP_SUB)
            # 含 = 且有数学符号，且不是流程图（→ 箭头串多个阶段）
            arrow_count = text.count("→") + text.count("->")
            looks_flow = arrow_count >= 2
            if (has_sign or has_supsub) and not looks_flow:
                suspicious_pre.append((path.name, text.strip()[:100]))

    print("=== 每课 <math> 块数 ===")
    for name, n in per_lesson.items():
        mark = "" if n else "  (无 MathML)"
        print(f"  {name}: {n}{mark}")
    print(f"总计 <math> 块: {total_math}")

    print("\n=== 良构性 ===")
    if malformed:
        print(f"❌ 发现 {len(malformed)} 个不良构 MathML：")
        for name, idx, err, snip in malformed:
            print(f"  {name} #[{idx}]: {err}\n     {snip}")
    else:
        print("✓ 所有 <math> 块均为良构 XML")

    print("\n=== 残留在 <pre> 里的疑似显示公式（应考虑转 MathML）===")
    if suspicious_pre:
        for name, snip in suspicious_pre:
            print(f"  {name}: {snip}")
    else:
        print("✓ 无 <pre> 残留显示型公式")

    return 1 if malformed else 0


if __name__ == "__main__":
    sys.exit(main())
