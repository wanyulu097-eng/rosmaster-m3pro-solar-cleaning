# 离线白框像素候选：研究版本

本例复用项目原有四边联合候选，通过本地图片产生像素候选或拒绝结果，不连接摄像头、ROS 或小车。它不能验证板面身份、物理尺寸、彩深配准、完整清洁区域或机械臂动作。

## 运行

需要 Python、NumPy 和 OpenCV（Python 包 `numpy`、`opencv-python`）。在仓库根目录运行：

```bash
python examples/white-frame-review/panel_joint_edges.py \
  --color examples/truncated-panel/panel-truncated-20261105.png \
  --output-dir output/white-frame-review
```

程序产生 JSON 和叠加图。检查 JSON 的 `decision`、`diagnostics.group_blocker`，不要只看画出的线。项目曾在 Python 3.14.3、OpenCV 4.10.0、NumPy 2.5.3 环境运行一次；不同环境可能影响线段检测结果。

## 已知结果和局限

对仓库中的真实出框帧，此版本的一次离线运行返回 `WHITE_FRAME_HYPOTHESIS_BLOCKED`，未输出四角；诊断显示没有满足完整边支持约束的闭合四边形。**这不证明程序识别出了真实截断原因。** 此图已用于项目开发，不是独立测试集。

该版本还曾在一张基本完整的已知对照帧上漏检。它只是一种依据亮暗线段和闭合几何的假设，尚不能区分所有非光伏板的矩形，也未经过覆盖多场景的独立验证。它没有经部署验证；请勿接入清洁控制。欢迎在 [Issue #1](../../issues/1) 讨论。

来源为项目本地 2026-10-31 的四边联合隔离候选，派生自项目中的白框审查工具；本文件的本地原始 SHA-256 为 `3adfe0dade1550ca6151e98f8770ec107124020d54ca08a2a4b52dacc496ef37`。NumPy 与 OpenCV 为独立第三方依赖，请遵守各自许可；本仓库的许可证尚待维护者确定。
