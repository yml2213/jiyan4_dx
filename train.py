import torch
from torch import nn
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from utils_size import auto_hw_from_ref
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设备选择：优先 CUDA，其次 Apple MPS，最后 CPU
def _has_mps() -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    try:
        return bool(mps is not None and hasattr(mps, "is_available") and mps.is_available())
    except Exception:
        return False

force_dev = os.getenv("FORCE_DEVICE")
if force_dev in {"cpu", "cuda", "mps"}:
    device = torch.device(force_dev)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif _has_mps():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

def _sync_device():
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            import torch.mps  # type: ignore
            torch.mps.synchronize()  # noqa: F401
    except Exception:
        pass

# 根据参考图像自适应输入大小（保证为 64 的倍数）
_h, _w, _gw, _gh = auto_hw_from_ref('images/1.jpg', multiple=64, default_hw=(192, 320))

# 网格划分（宽×高），等于输入尺寸除以 64
needJj = [_gw, _gh]
# 图片压缩为的大小（h, w）——transforms.Resize 约定顺序为 (h, w)
tpxz = (_h, _w)

def euclidean_distance(p1, p2):
    '''
    计算两个点的欧式距离
    '''
    x1, y1 = p1
    x2, y2 = p2
    return torch.sqrt((x2-x1)**2 + (y2-y1)**2)
class BBox:
    def __init__(self, xe, ye, re, be,dd = 0):
        '''
        定义框，左上角及右下角坐标
        '''
        if dd == 1:
            self.x, self.y, self.r, self.b = xe, ye, re, be
        else:
            if re/2 >xe and 1==0:
                x = 0
            else:
                x = xe - re/2

            if be/2 > ye and 1==0:
                y = 0
            else:
                y = ye - be/2

            if xe + re/2 > 1 and 1==0:
                r = 1
            else:
                r = xe + re/2
            if ye + be / 2 > 1 and 1==0:
                b = 1
            else:
                b = ye + be / 2

            self.x, self.y, self.r, self.b = x, y, r, b

    def __xor__(self, other):
        '''
        计算box和other的IoU
        '''
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)

    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross
        return union

    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        xmax = min(self.r, other.r)
        ymax = min(self.b, other.b)
        xmin = max(self.x, other.x)
        ymin = max(self.y, other.y)
        cross_box = BBox(xmin, ymin, xmax, ymax, 1)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0
        return cross_box.area

    def boundof(self, other):
        '''
        计算box和other的边缘外包框，使得2个box都在框内的最小矩形
        '''
        xmin = min(self.x, other.x)
        ymin = min(self.y, other.y)
        xmax = max(self.r, other.r)
        ymax = max(self.b, other.b)
        return BBox(xmin, ymin, xmax, ymax, 1)

    def center_distance(self, other):
        '''
        计算两个box的中心点距离
        '''
        return euclidean_distance(self.center, other.center)

    def bound_diagonal_distance(self, other):
        '''
        计算两个box的bound的对角线距离
        '''
        bound = self.boundof(other)
        return euclidean_distance((bound.x, bound.y), (bound.r, bound.b))

    @property
    def center(self):
        return (self.x + self.r) / 2, (self.y + self.b) / 2

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.r - self.x  # + 1

    @property
    def height(self):
        return self.b - self.y  # + 1

with open('./classes.txt',encoding='utf-8') as f:
    t = f.read().split('\n')
alllb = len(t)
class getData(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []
        path = './labels'
        img_dir = './images'
        for i in os.listdir(path):
            if not i.lower().endswith('.txt'):
                continue
            base = os.path.splitext(i)[0]
            # 同时支持 .png/.jpg/.jpeg（大小写均可）
            cands = [
                os.path.join(img_dir, base + ext)
                for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
            ]
            img_path = None
            for p in cands:
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                # 找不到同名图片则跳过该样本
                continue
            self.data.append([img_path, os.path.join(path, i)])
        # 统一排序，便于范围选择（按标签文件名排序）
        self.data.sort(key=lambda x: x[1])

        # 可选：通过环境变量筛选子集
        # LABEL_RANGE: 形如 "1-10"（1 基）
        lr = os.getenv('LABEL_RANGE')
        if lr:
            try:
                a, b = lr.strip().split('-', 1)
                a_i = max(1, int(a))
                b_i = int(b)
                self.data = self.data[a_i - 1:b_i]
            except Exception:
                pass

        # LIMIT_N: 仅取前 N 个样本
        limit_n = os.getenv('LIMIT_N')
        if limit_n:
            try:
                n = int(limit_n)
                if n > 0:
                    self.data = self.data[:n]
            except Exception:
                pass
        self.jk = len(self.data)
        self.tpcl = transforms.Compose([
            transforms.Resize(tpxz),
            transforms.ToTensor()
        ])
        self.alllb = alllb


    def pdisIn(self,x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) <= min(x2, x4) and max(y1, y3) <= min(y2, y4):
            return True
        else:
            return False

    def niou(self,rec1, rec2):
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        # S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / S2

    def __getitem__(self, item):
        dt = self.data[item]
        with open(dt[1], encoding='utf-8') as lp:
            kj = lp.read()
        # 解析标签为浮点数列表，跳过空行/无效行
        rows = [ln.strip() for ln in kj.split('\n') if ln.strip()]
        h = []  # list[list[float]]
        for ln in rows:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                nums = [float(x) for x in parts]
                h.append(nums)
            except Exception:
                continue
        imge = Image.open(dt[0]).convert('RGB')
        img = self.tpcl(imge).permute(0, 2,1)
        xz = 1 / needJj[0]
        yz = 1 / needJj[1]

        target = torch.zeros((needJj[0],needJj[1],9,6)).to(device)

        for gx in range(needJj[0]):
            for gy in range(needJj[1]):
                sj = [xz*gx, yz*gy, xz*gx+xz, yz*gy+yz]
                for _, ko in enumerate(h):
                    ges = 0  # 与原逻辑一致，固定写入候选槽 0
                    kol = [ko[1]-ko[3]/2, ko[2]-ko[4]/2, ko[1]+ko[3]/2, ko[2]+ko[4]/2]
                    lpijk = self.niou([sj[0], sj[1], sj[2], sj[3]], [kol[0], kol[1], kol[2], kol[3]])
                    if self.pdisIn(sj[0], sj[1], sj[2], sj[3], kol[0], kol[1], kol[2], kol[3]) and lpijk > 0.1 and lpijk > target[gx, gy, ges, 4]:
                        target[gx, gy, ges, 0] = ko[1]
                        target[gx, gy, ges, 1] = ko[2]
                        target[gx, gy, ges, 2] = ko[3]
                        target[gx, gy, ges, 3] = ko[4]
                        target[gx, gy, ges, 4] = lpijk
                        # target[gx, gy, ges, 5:] = 0
                        target[gx, gy, ges, 5] = 1.0
                        # target[gx, gy, ges, int(ko[0]) + 6] = 1
                    # else:
                    #     target[x, i, ges, 4:] = 0
                        # target[x,i,ges,5] = 0
                    # break


        return img.to(device), target


    def __len__(self):
        return self.jk




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
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
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



class mbLoss(nn.Module):
    def __init__(self):
        super(mbLoss, self).__init__()
        self.jcs = nn.BCEWithLogitsLoss()

    def CIoU(self,a, b):
        v = 4 / (torch.pi ** 2) * (torch.atan(a.width / a.height) - torch.atan(b.width / b.height)) ** 2
        iou =self.IoU(a, b)
        alpha = v / (1 - iou + v)
        return 1 - (self.DIoU(a, b) - alpha * v), iou

    def DIoU(self,a, b):
        d = a.center_distance(b)
        c = a.bound_diagonal_distance(b)
        return self.IoU(a, b) - (d ** 2) / (c ** 2)
    def IoU(self,a, b):
        return a ^ b

    def forward(self,out, target):

        allloss = 0
        zsd = 0
        jbb = 0
        qit = 0
        huhuh = 0
        jsq = 0
        for bash in range(target.shape[0]):
            for xwz in range(target.shape[1]):
                zxdwx = (1 / target.shape[1]) * xwz + (1 / target.shape[1]) / 2
                for ywz in range(target.shape[2]):
                    zxdwy = (1/target.shape[2])*ywz + (1/target.shape[2])/2
                    dt = out[bash, xwz, ywz, :, :]
                    for qub in range(target.shape[3]):
                        st = target[bash, xwz, ywz, qub,:]
                        if st[5] > 0.8:
                            for jk in range(dt.shape[0]):
                                a = BBox(st[0], st[1], st[2], st[3])
                                b = BBox((dt[jk][0]-0.5) + zxdwx, (dt[jk][1]-0.5) +zxdwy, (dt[jk][2]-0.5) + 1/target.shape[1], (dt[jk][3]-0.5) + 1/target.shape[2] )
                                los, iou =  self.CIoU(a, b)
                                allloss += los
                                jbb += iou
                                jsq += 1
                                zsd += (1- dt[jk][4]) ** 2
                                allloss += (1- dt[jk][4])
                                huhuh += (1- dt[jk][4])

                        else:
                            for jk in range(dt.shape[0]):
                                zsd += dt[jk][4] ** 2

                                qit += 1
                        break

        return allloss/ jsq + zsd / (jsq+qit), jbb/ jsq, huhuh / (jsq)





data = getData()

mymodo = mubModu()
# mymodo = torch.load('./mox2.pth')
mymodo.to(device)
meLoss = mbLoss()
optm = torch.optim.Adam(mymodo.parameters(),lr=0.001)
maxLoss = 10000

# 允许通过环境变量控制训练轮数与批大小（默认与原始脚本一致）
epochs = int(os.getenv("EPOCHS", "100"))
batch_size = int(os.getenv("BATCH_SIZE", "30"))

ep_bar = tqdm(range(epochs), desc="总进度", position=0, leave=True)
for i in ep_bar:

    sx = 0
    cs = 0
    csl = 0
    zxdss = 0
    # 可调的 DataLoader 参数（用于性能调优）
    num_workers = int(os.getenv("NUM_WORKERS", "0"))
    prefetch_factor = int(os.getenv("PREFETCH_FACTOR", "2"))
    dl_kwargs = dict(shuffle=True, batch_size=batch_size)
    if num_workers > 0:
        dl_kwargs.update(dict(num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor))
    datae = DataLoader(data, **dl_kwargs)

    profile = os.getenv("PROFILE", "0") == "1"
    t_prev_end = time.perf_counter()
    acc_load = acc_fwd = acc_loss = acc_bwd = acc_step = 0.0

    # 迭代进度条：放在第二行，完成后不保留，避免结束时出现 0%
    total_steps = len(datae) if hasattr(datae, "__len__") else None
    with tqdm(datae, total=total_steps, desc=f"第{i+1}/{epochs}轮", position=1, leave=False) as datad:
        for img, tar in datad:
            t0 = time.perf_counter()
            acc_load += max(0.0, t0 - t_prev_end)
            optm.zero_grad()
            _sync_device();
            t1 = time.perf_counter()
            out = mymodo(img)
            _sync_device();
            t2 = time.perf_counter()
            loss, ub, zxd = meLoss(out, tar)
            _sync_device();
            t3 = time.perf_counter()
            loss.backward()
            _sync_device();
            t4 = time.perf_counter()
            optm.step()
            _sync_device();
            t5 = time.perf_counter()
            acc_fwd += t2 - t1
            acc_loss += t3 - t2
            acc_bwd += t4 - t3
            acc_step += t5 - t4
            t_prev_end = t5
            sx += loss.item()
            cs += ub.item()
            zxdss += zxd.item()
            csl += 1

            datad.set_description("训练 loss {:.6f} epch {} iou {:.6f} zxd {:.6f}".format(sx / max(csl,1), i, cs / max(csl,1), zxdss / max(csl,1)))

    if profile:
        total = acc_load + acc_fwd + acc_loss + acc_bwd + acc_step
        print("[PROFILE] load={:.3f}s fwd={:.3f}s loss={:.3f}s bwd={:.3f}s step={:.3f}s total={:.3f}s".format(
            acc_load, acc_fwd, acc_loss, acc_bwd, acc_step, total
        ))

    # 更新总进度条的摘要信息
    if csl > 0:
        ep_bar.set_postfix(loss=f"{sx/csl:.6f}", iou=f"{cs/csl:.6f}", zxd=f"{zxdss/csl:.6f}")

    if csl > 0 and sx / csl < maxLoss:
        torch.save(mymodo, './mox2.pth')
        print("保存模型===》")
        maxLoss = sx / csl

# 训练结束后输出完成提示，确保“总进度”达到 100%
tqdm.write("训练完成：{} 轮，最优 loss = {:.6f}".format(epochs, maxLoss))




