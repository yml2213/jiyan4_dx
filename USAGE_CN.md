项目使用说明（中文）

概览
- 目标：训练一个轻量级的目标检测模型，导出为 ONNX，并在图片上画框推理。
- 关键脚本与文件：
  - `train.py`：训练模型，输出 `mox2.pth`。
  - `bconnx.py`：将 `mox2.pth` 导出为 `sbkuan.onnx`。
  - `huak.py`：使用 ONNXRuntime 推理并在图片上绘制检测框。
  - `images/`：图片目录（训练期需要 `.png`；推理期 `.jpg/.png` 均可）。
  - `labels/`：标签目录（YOLO 文本格式）。
  - `classes.txt`：类别名称清单（当前模型未输出类别，仅保留以备扩展）。

环境准备
- Python 3.9+（建议 3.11）。
- 安装依赖：
  - `python3 -m pip install -r requirements.txt`
- 设备：自动选择 `cuda` 或 `cpu`，GPU 可加速训练。

数据准备
- 目录组织：
  - 图片放到 `images/`，标签放到 `labels/`，两者同名，如 `images/1.png` 对应 `labels/1.txt`。
- 标签格式（YOLO 风格，数值均为 0~1 的相对比例）：
  - 每行：`class cx cy w h`
  - 示例：`4 0.1015625 0.24722222222222223 0.19062500000000002 0.35`
- 训练输入仅按 `.png` 查找图片：
  - 若你手头是 `.jpg`，请复制或转换为同名 `.png`：
    - macOS/Linux：`for f in images/*.jpg; do cp "$f" "${f%.jpg}.png"; done`

快速开始（最短路径）
1) 安装依赖：`python3 -m pip install -r requirements.txt`
2) 准备数据：将图片与同名标签分别放入 `images/`、`labels/`（训练图为 `.png`）。
3) 训练模型（快速试跑 1 轮）：`EPOCHS=1 BATCH_SIZE=8 python3 -u train.py`
   - 正式训练：`python3 -u train.py`（默认 100 轮、批大小 30）。
   - 训练完成会在根目录生成 `mox2.pth`。
4) 导出 ONNX：`python3 -u bconnx.py`，将生成 `sbkuan.onnx`。
5) 推理画框：
   - 使用默认示例图：`python3 -u huak.py`
   - 指定图片：`python3 -u huak.py --img images/2.jpg --out runs/out.png`
   - 显示窗口（本机有 GUI）：`python3 -u huak.py --img images/2.jpg --show`

训练说明
- 输入尺寸：`(3, 320, 192)`（与代码内张量维序一致）。
- 网格划分：`needJj = [5, 3]`（高×宽），每格 2 个候选框。
- 模型输出：每格 2 个框，共 5 个值 `[cx, cy, w, h, conf]`，范围 0~1。
- 环境变量（已添加）
  - `EPOCHS`：训练轮数，默认 `200`。
  - `BATCH_SIZE`：批大小，默认 `30`。
  - 示例：`EPOCHS=50 BATCH_SIZE=16 python3 -u train.py`
- 产物：
  - `mox2.pth`：训练得到的 PyTorch 模型（按最优 loss 自动保存）。

导出与推理
- 导出：`python3 -u bconnx.py` → 生成 `sbkuan.onnx`。
- 推理脚本：`huak.py`（ONNXRuntime）
  - 自动在 `images/1.jpg|png` → `1_.png` 上测试；
  - 或用参数：`--img <图片路径> --out <输出路径> [--show]`。
  - 置信度阈值在代码中为 `0.9`，可根据需要下调。

使用你的图片做识别（步骤）
1) 将图片放入仓库：如 `images/my.jpg`。
2) 若未导出 ONNX：`python3 -u bconnx.py`。
3) 执行推理：`python3 -u huak.py --img images/my.jpg --out runs/my_out.png`。
4) 查看结果：打开 `runs/my_out.png`，红色矩形即检测框。

从零开始构建自己的识别
1) 数据采集：在目标场景多采样、不同光照/角度/遮挡。
2) 数据标注：用 LabelImg/Labelme/Roboflow 标注并导出 YOLO 文本格式。
3) 数据整理：按同名放入 `images/` 与 `labels/`，训练图准备为 `.png`。
4) 训练：`python3 -u train.py` 或控制轮数/批大小。
5) 导出：`python3 -u bconnx.py`。
6) 推理联调：`python3 -u huak.py --img ...`，根据效果调阈值、增广或加数据重训。

项目结构（简要）
- `train.py`：训练逻辑与损失（框 + 置信度），可通过环境变量控制轮数/批大小。
- `bconnx.py`：加载 `mox2.pth` 并导出 ONNX。
- `huak.py`：加载 `sbkuan.onnx`，前处理、推理、NMS/合并逻辑与画框。
- `images/`、`labels/`：样例数据。
- `classes.txt`：类名清单（当前未用于分类头和损失）。

常见问题（FAQ）
- 训练阶段找不到图片：训练脚本仅查找 `.png`，请将训练图片转为 `.png` 并与标签同名。
- 不能弹窗显示：在无 GUI 环境不要使用 `--show`，查看保存的输出图即可。
- 没有类别输出：当前模型只学习框与置信度；若需要类别，请扩展网络头部与损失，并在推理时解码类别概率与阈值。
- 速度/设备：GPU 会更快；CPU 可正常运行但速度较慢。

扩展建议
- 同时支持 `.jpg/.png` 的训练数据自动匹配。
- 增加分类分支与可视化类别文字（读取 `classes.txt`）。
- 数据增强与更深的主干网络以提升召回率与定位精度。

