项目使用说明（中文）

概览

- 目标：训练一个轻量级的目标检测模型，导出为 ONNX，并在图片上画框推理。
- 关键脚本与文件：
  - `train.py`：训练模型，输出 `mox2.pth`。
  - `bconnx.py`：将 `mox2.pth` 导出为 `sbkuan.onnx`。
  - `huak.py`：使用 ONNXRuntime 推理并在图片上绘制检测框。
  - `images/`：图片目录（训练/推理均支持 `.jpg/.png/.jpeg`）。
  - `labels/`：标签目录（YOLO 文本格式）。
  - `classes.txt`：类别名称清单（当前模型未输出类别，仅保留以备扩展）。

环境准备

- Python 3.9+（建议 3.11）。
- 安装依赖：
  - `python3 -m pip install -r requirements.txt`
- 设备：自动选择 `cuda` 或 `cpu`，GPU 可加速训练。

使用 uv（可选）

- 目标：用 `uv` 管理 Python 版本与虚拟环境，固定到 3.10.x。
- 安装 uv：
  - macOS（Homebrew）：`brew install uv`
  - 通用脚本：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 安装 Python：`uv python install 3.10.13`
- 创建虚拟环境（项目根目录）：`uv venv --python 3.10.13`
- 激活（macOS/Linux）：`source .venv/bin/activate`
- 安装依赖：`uv pip install -r requirements.txt`
- 运行示例：
  - 训练试跑：`EPOCHS=1 BATCH_SIZE=8 uv run python -u train.py`
  - 导出 ONNX：`uv run python -u bconnx.py`
  - 推理画框：`uv run python -u huak.py --img images/1.jpg`

数据准备

- 目录组织：
  - 图片放到 `images/`，标签放到 `labels/`，两者同名，如 `images/1.png` 或 `images/1.jpg` 对应 `labels/1.txt`。
- 标签格式（YOLO 风格，数值均为 0~1 的相对比例）：
  - 每行：`class cx cy w h`
  - 示例：`4 0.1015625 0.24722222222222223 0.19062500000000002 0.35`
- 训练输入已同时支持 `.png/.jpg/.jpeg` 自动匹配，无需转换。

快速开始（最短路径）

1. 安装依赖：`python3 -m pip install -r requirements.txt`
2. 准备数据：将图片与同名标签分别放入 `images/`、`labels/`。
3. 训练模型（快速试跑 1 轮）：`EPOCHS=1 BATCH_SIZE=8 python3 -u train.py`
   - 正式训练：`python3 -u train.py`（默认 100 轮、批大小 30）。
   - 训练完成会在根目录生成 `mox2.pth`。
4. 导出 ONNX：`python3 -u bconnx.py`，将生成 `sbkuan.onnx`。
5. 推理画框：
   - 使用默认示例图：`python3 -u huak.py`
   - 指定图片：`python3 -u huak.py --img images/2.jpg --out runs/out.png`
   - 显示窗口（本机有 GUI）：`python3 -u huak.py --img images/2.jpg --show`

按子集训练（仅前 N 张或范围）

- 通过环境变量筛选训练子集：
  - 仅前 N 张：`LIMIT_N=10 EPOCHS=1 python3 -u train.py`
  - 指定范围（按标签名排序，1 基）：`LABEL_RANGE=1-10 EPOCHS=1 python3 -u train.py`
  - 可与 `BATCH_SIZE` 等参数组合使用。

训练示例（脚本）

- 仅用第 1~10 张训练：`examples/train_range_1_10.sh`
- 训练 600 轮、每批 30 张：`examples/train_600ep_bs30.sh`
- 计算当前“每轮步数”(steps/epoch)：`python examples/steps_per_epoch.py`
  - 步数公式：`ceil(样本数 / BATCH_SIZE)`；本项目 `DataLoader` 默认不丢弃最后一个不满批次。
  - 受环境变量 `LABEL_RANGE` 与 `LIMIT_N` 影响，可先筛选再计算。

训练说明

- 输入尺寸（自适应）：训练/导出/推理统一根据参考图 `images/1.jpg` 自动确定，按 64 的倍数就近取整；若读取失败则回退到 320×192。
- 网格划分：由输入尺寸推导，`needJj = [宽/64, 高/64]`（宽 × 高），每格 2 个候选框。
- 模型输出：每格 2 个框，共 5 个值 `[cx, cy, w, h, conf]`，范围 0~1。
- 环境变量（已添加）
  - `EPOCHS`：训练轮数，默认 `100`。
  - `BATCH_SIZE`：批大小，默认 `30`。
  - 示例：`EPOCHS=50 BATCH_SIZE=16 python3 -u train.py`
- 产物：
  - `mox2.pth`：训练得到的 PyTorch 模型（按最优 loss 自动保存）。

导出与推理

- 导出：`python3 -u bconnx.py` → 生成 `sbkuan.onnx`。
- 推理脚本：`huak.py`（ONNXRuntime）
  - 自动在 `images/1.jpg|png` → `1_.png` 上测试；
  - 或用参数：`--img <图片路径> --out <输出路径> [--show]`。
  - 执行提供器（加速后端）选择：`--ep auto|cpu|cuda|coreml`，默认 `auto`（优先 CUDA 或 Apple CoreML）。
  - 置信度阈值在代码中为 `0.9`，可根据需要下调。

Apple Silicon（MPS）

- 训练/导出：脚本会优先选择 CUDA → MPS → CPU，并打印 `Using device: ...`。
- 建议在 macOS 上启用算子回退（缺失算子回退到 CPU）：`export PYTORCH_ENABLE_MPS_FALLBACK=1`。
- 推理：若需使用 Apple GPU，请指定 CoreML 执行提供器（若可用）：`python -u huak.py --img images/1.jpg --ep coreml`；若不可用会自动回退到 CPU。

使用你的图片做识别（步骤）

1. 将图片放入仓库：如 `images/my.jpg`。
2. 若未导出 ONNX：`python3 -u bconnx.py`。
3. 执行推理：`python3 -u huak.py --img images/my.jpg --out runs/my_out.png`。
4. 查看结果：打开 `runs/my_out.png`，红色矩形即检测框。

从零开始构建自己的识别

1. 数据采集：在目标场景多采样、不同光照/角度/遮挡。
2. 数据标注：用 LabelImg/Labelme/Roboflow 标注并导出 YOLO 文本格式。
3. 数据整理：按同名放入 `images/` 与 `labels/`，训练图准备为 `.png`。
4. 训练：`python3 -u train.py` 或控制轮数/批大小。
5. 导出：`python3 -u bconnx.py`。
6. 推理联调：`python3 -u huak.py --img ...`，根据效果调阈值、增广或加数据重训。

项目结构（简要）

- `train.py`：训练逻辑与损失（框 + 置信度），可通过环境变量控制轮数/批大小。
- `bconnx.py`：加载 `mox2.pth` 并导出 ONNX。
- `huak.py`：加载 `sbkuan.onnx`，前处理、推理、NMS/合并逻辑与画框。
- `images/`、`labels/`：样例数据。
- `classes.txt`：类名清单（当前未用于分类头和损失）。

常见问题（FAQ）

- 训练阶段找不到图片：脚本已支持 `.jpg/.png/.jpeg`，请检查是否与标签同名（不含扩展名一致）。
- 不能弹窗显示：在无 GUI 环境不要使用 `--show`，查看保存的输出图即可。
- 没有类别输出：当前模型只学习框与置信度；若需要类别，请扩展网络头部与损失，并在推理时解码类别概率与阈值。
- 速度/设备：GPU 会更快；CPU 可正常运行但速度较慢。

扩展建议

- 同时支持 `.jpg/.png` 的训练数据自动匹配。
- 增加分类分支与可视化类别文字（读取 `classes.txt`）。
- 数据增强与更深的主干网络以提升召回率与定位精度。

FORCE_DEVICE=cpu EPOCHS=200 BATCH_SIZE=30 LABEL_RANGE=1-100 PROFILE=1 PY=.venv/bin/python bash examples/train_range_1_10.sh
