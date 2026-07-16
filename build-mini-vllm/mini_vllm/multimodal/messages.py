"""结构化多模态消息（Lesson 19）。

一条消息有角色（role）和一串内容项（content），内容项可以是文本、图片或视频。
用普通 dict 表示，便于放进 JSON workload：

    {"role": "user", "content": [
        {"type": "text",  "text": "这张图里有什么？"},
        {"type": "image", "image": <pixels 或 {"path": ...} 或 {"synth": {...}}>},
    ]}

关键区分（Lesson 19 的核心）：图片路径/像素**不是**模型的输入 embedding；
chat template 不编码像素；tokenizer 不处理像素——它们是不同阶段。
"""

from __future__ import annotations


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def image(pixels_or_ref) -> dict:
    return {"type": "image", "image": pixels_or_ref}


def video(ref) -> dict:
    return {"type": "video", "video": ref}


def user(*content) -> dict:
    return {"role": "user", "content": list(content)}


def system(*content) -> dict:
    return {"role": "system", "content": list(content)}
