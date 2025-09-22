安装 uv（任选其一）
macOS（Homebrew）：brew install uv
通用（官方脚本）：curl -LsSf https://astral.sh/uv/install.sh | sh
安装 Python 3.10.x（建议固定补丁版）
uv python install 3.10.13
可查已装版本：uv python list
创建并使用 uv 虚拟环境（在项目根目录）
创建：uv venv --python 3.10.13
激活（macOS/Linux）：source .venv/bin/activate
激活（Windows）：.venv\Scripts\activate
安装依赖到虚拟环境
uv pip install -r requirements.txt

python -V 应输出 Python 3.10.13
python - <<'PY'\nimport torch, onnxruntime as ort; print('torch', torch.**version**); print('onnxruntime', ort.**version**)\nPY
