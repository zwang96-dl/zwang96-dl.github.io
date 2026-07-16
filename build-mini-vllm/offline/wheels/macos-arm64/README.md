# offline/wheels/macos-arm64/

放置为**锁定环境**（macOS 14+ / Apple Silicon / Python 3.11 / arm64）准备的 wheel。

**核心课程不需要任何 wheel**（纯标准库实现），因此本目录默认只含本说明。
`offline/manifests/wheels.json` 的 `required` 为 `[]`，`verify-offline-bundle` 会据此通过。

若要加入可选的 PyTorch 加速，请按 `../../INSTALL_OFFLINE.md` 的「可选加速」章节，
把 wheel 下载到这里，并在 `offline/manifests/wheels.json` 登记 filename 与 sha256。
