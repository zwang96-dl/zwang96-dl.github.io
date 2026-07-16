"""Chat Template —— 把结构化消息渲染成「文本段 + 媒体占位」的序列（Lesson 19）。

chat template 的职责：加入角色标记、把内容项排成线性顺序、在媒体处放一个**占位标记**。
它**不**编码像素，也**不**决定 visual token 数——那是 processor / encoder 的事。

render() 返回 segments 列表：
    ("text", "<某段文本>")
    ("media", {"modality": "image"|"video", "ref": <媒体引用>})
以及一个便于调试的 rendered 字符串（媒体处显示 <image>/<video>）。
"""

from __future__ import annotations


class MultiModalChatTemplate:
    ROLE_PREFIX = {"system": "System: ", "user": "User: ", "assistant": "Assistant: "}

    def render(self, messages: list[dict]):
        segments = []
        rendered = []
        for msg in messages:
            prefix = self.ROLE_PREFIX.get(msg.get("role", "user"), "")
            if prefix:
                segments.append(("text", prefix))
                rendered.append(prefix)
            for item in msg["content"]:
                t = item["type"]
                if t == "text":
                    segments.append(("text", item["text"]))
                    rendered.append(item["text"])
                elif t == "image":
                    segments.append(("media", {"modality": "image", "ref": item["image"]}))
                    rendered.append("<image>")
                elif t == "video":
                    segments.append(("media", {"modality": "video", "ref": item["video"]}))
                    rendered.append("<video>")
                else:
                    raise ValueError(f"未知内容类型：{t}")
            segments.append(("text", "\n"))
            rendered.append("\n")
        return segments, "".join(rendered)
