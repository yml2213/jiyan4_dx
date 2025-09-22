# 把pth模型转onnx类型


import torchvision.models as models
import torch
from torch import nn
from utils_size import auto_hw_from_ref

# 设备选择：优先 CUDA，其次 Apple MPS，最后 CPU；交互确认是否使用 GPU/MPS
def _has_mps() -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    try:
        return bool(mps is not None and hasattr(mps, "is_available") and mps.is_available())
    except Exception:
        return False

if torch.cuda.is_available():
    if input("检测到 CUDA，可使用 GPU 运行，是否启用？(y/n): ").strip().lower() == 'y':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
elif _has_mps():
    if input("检测到 Apple GPU (MPS)，是否启用？(y/n): ").strip().lower() == 'y':
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
print("Using device:", device)

with open('./classes.txt',encoding='utf-8') as f:
    t = f.read().split('\n')
alllb = len(t)

class mubModu(nn.Module):
    def __init__(self):
        super(mubModu, self).__init__()
        self.ks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=12, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        d2 = self.ks(x)
        d2 = d2.permute(0, 2, 3, 1)
        d2 = d2.reshape((d2.shape[0], d2.shape[1], d2.shape[2], 2, 5))
        out = d2.squeeze(0)
        return out



try:
    mymodo = torch.load('./mox2.pth', map_location=device)
    mymodo.to(device)
    mymodo.eval()

    input_names = ['input']
    output_names = ['output']

    # 自适应输入大小：保持与训练一致（宽、高均为 64 的倍数）
    _h, _w, _gw, _gh = auto_hw_from_ref('images/1.jpg', multiple=64, default_hw=(192, 320))
    x = torch.randn(1, 3, _w, _h).to(device)

    torch.onnx.export(mymodo, x, 'sbkuan.onnx', input_names=input_names, output_names=output_names, verbose=True)
    print("成功把画框模型转onnx")
except:
    pass



