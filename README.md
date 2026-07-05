# 我的文档库 📚

用 GitHub Pages 搭建的个人文档主页。**每个项目一个文件夹,里面放一个 `index.html`**。
推送到 GitHub 后,云端自动扫描并生成带**搜索、分类、深色模式**的门户主页,
访问 `https://zwang96-dl.github.io` 即可。

## 目录结构

```
.
├── index.html                  # 自动生成的门户主页(别手动改,每次构建会覆盖)
├── build.py                    # 扫描各项目文件夹、生成 index.html 的脚本
├── .nojekyll                   # 关闭 GitHub 的 Jekyll 处理(别删)
├── 欢迎/
│   └── index.html              # ← 一个项目 = 一个文件夹,里面就叫 index.html
├── vllm-paged-attention/
│   └── index.html
├── .github/workflows/
│   └── deploy.yml              # push 后自动构建 + 部署
└── README.md
```

> 规则:根目录下**每个含 `index.html` 的文件夹**都会变成主页上的一张卡片。
> 卡片标题默认就是文件夹名。不含 `index.html` 的文件夹、以 `.` 开头的文件夹会自动忽略。

---

## 首次部署(只做一次)

**1. 改个人信息**：打开 `build.py`,修改顶部的 `SITE`(名字、简介、GitHub 链接)。

**2. 在 GitHub 建仓库**：仓库名必须叫 `zwang96-dl.github.io`(全小写),设为 **Public**。

**3. 推送代码**：

```bash
git init
git add .
git commit -m "init: 我的文档库"
git branch -M main
git remote add origin git@github.com:zwang96-dl/zwang96-dl.github.io.git
git push -u origin main
```

**4. 打开 Pages 部署**(关键一步)：
到仓库 **Settings → Pages → Build and deployment → Source**,选择 **GitHub Actions**(不是 "Deploy from a branch")。

**5. 等约 1 分钟**,在仓库 **Actions** 标签页看到绿色对勾后,访问：

```
https://zwang96-dl.github.io
```

---

## 日常使用:加一个新项目

```bash
# 1. 建一个项目文件夹,把文档存成里面的 index.html
mkdir cuda-warp-primitives
# ...把你写好的 HTML 存为 cuda-warp-primitives/index.html,配图等资源也放这个文件夹

# 2. 提交推送,剩下的全自动
git add .
git commit -m "新增:CUDA warp 原语笔记"
git push
```

推送后无需其它操作,主页自动多出一张卡片。
该项目的地址就是 `https://zwang96-dl.github.io/cuda-warp-primitives/`。

> 文件夹名建议用**英文 + 连字符**(如 `paged-attention`),URL 最干净;用中文也能正常工作。

---

## 可选:让卡片显示更精致

在项目 `index.html` 的 `<head>` 里加这些 meta(都可选):

| meta 标签 | 作用 | 不填时的兜底 |
|-----------|------|--------------|
| `doc-title` | 卡片标题 | 取 `<title>`,再没有就取**文件夹名** |
| `doc-category` | 分类(加了才会出现分类过滤条) | 无分类 |
| `doc-description` | 卡片简介 | 空 |
| `doc-date` | 日期(用于排序,新的在前) | 取 git 提交日期 |
| `doc-tags` | 标签(逗号分隔) | 无 |

```html
<meta name="doc-category"    content="推理优化">
<meta name="doc-description" content="warp 级原语与 shuffle 指令速查">
<meta name="doc-tags"        content="GPU,并行,速查">
```

---

## 本地预览

```bash
python3 build.py                  # 生成最新 index.html
python3 -m http.server 8000       # 浏览器打开 http://localhost:8000
```
