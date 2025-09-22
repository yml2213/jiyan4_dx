from typing import Any, cast
import cv2 as _cv2
from PIL import Image
import time
import onnxruntime as ort
import numpy as np
import os
import argparse
from utils_size import auto_hw_from_ref


cv2 = cast(Any, _cv2)  # silence type checkers; cv2 has dynamic attributes


class getTpInfo:
    def __init__(self, onnx_path: str = './sbkuan.onnx', providers: list | None = None):
        with open('./classes.txt', encoding='utf-8') as f:
            t = f.read().split('\n')
        self.alllb = t
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f'未找到模型文件: {onnx_path}，请先运行 bconnx.py 生成。')
        # 自动选择可用的推理后端（EP）。优先 CUDA、CoreML（Apple）、最后 CPU。
        available = ort.get_available_providers()
        if providers is None:
            picked = []
            for cand in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"):
                if cand in available:
                    picked.append(cand)
            if not picked:
                picked = ["CPUExecutionProvider"]
        else:
            # 仅保留可用的 provider，避免初始化报错
            picked = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self.providers = picked
        self.mymodo = ort.InferenceSession(onnx_path, providers=self.providers)
        print(f"ONNXRuntime providers: {self.providers} (available: {available})")

    def bbbiou(self, rec1, rec2):
        if self.pdisIn(rec1[0], rec1[1], rec1[2], rec1[3], rec2[0], rec2[1], rec2[2], rec2[3]) is False:
            return 0
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S2 + S1 - S_cross)

    def pdisIn(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) <= min(x2, x4) and max(y1, y3) <= min(y2, y4):
            return True
        else:
            return False

    def hetInfo(self, out):
        out = out[0]
        lzx = 1 / out.shape[0]
        lzy = 1 / out.shape[1]
        kd = []

        for i in range(out.shape[0]):
            zxdwx = lzx * i + lzx / 2
            for i2 in range(out.shape[1]):
                zxdwy = lzy * i2 + lzy / 2
                for k in range(out.shape[2]):
                    if out[i, i2, k, 4] > 0.9:
                        zxx = (out[i, i2, k, 0] - 0.5) + zxdwx
                        zxy = (out[i, i2, k, 1] - 0.5) + zxdwy
                        zxk = (out[i, i2, k, 2] - 0.5) + lzx
                        zxg = (out[i, i2, k, 3] - 0.5) + lzy
                        l = [
                            zxx - zxk / 2,
                            zxy - zxg / 2,
                            zxx + zxk / 2,
                            zxy + zxg / 2,
                            out[i, i2, k, 4],
                        ]
                        isokk = 1
                        for idx, ds in enumerate(kd):
                            if self.bbbiou([l[0], l[1], l[2], l[3]], [ds[0], ds[1], ds[2], ds[3]]) < 0.1:
                                continue
                            else:
                                isokk = 0
                                if ds[4] < l[4]:
                                    kd[idx] = l
                        if isokk == 1:
                            kd.append(l)

        return kd

    def getimage(self, path):
        imge = Image.open(path).convert('RGB')
        # 与训练/导出保持一致的自适应尺寸
        _h, _w, _, _ = auto_hw_from_ref('images/1.jpg', multiple=64, default_hw=(192, 320))
        # Pillow 10+ 推荐使用 Resampling 枚举；旧别名 Image.BILINEAR 仍可用但在类型检查下告警
        dst = imge.resize((_w, _h), Image.Resampling.BILINEAR)
        dst.save('./l.jpg')
        dst = np.array(dst).astype(np.float32) / 255
        img = dst.transpose(2, 1, 0).reshape((1, 3, _w, _h))
        return img, imge

    def shibie(self, imgpa, show: bool = False, save_path: str = './1_.png'):
        tp, imge = self.getimage(imgpa)
        d = time.time()
        kuane = self.mymodo.run(None, {self.mymodo.get_inputs()[0].name: tp})
        print('识别用时', time.time() - d, '秒')

        kuan = self.hetInfo(kuane)  # 获取框的信息

        tp = cv2.imread(imgpa)
        y = tp.shape[0]
        x = tp.shape[1]
        for idx, i in enumerate(kuan):
            cv2.rectangle(tp, (int(i[0] * x), int(i[1] * y)), (int(i[2] * x), int(i[3] * y)), (0, 0, 255), 2)
        cv2.imwrite(save_path, tp)
        print(f'已保存: {save_path}')
        if show:
            try:
                cv2.imshow('image', tp)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
        return save_path, kuan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX 目标检测（画框）推理')
    parser.add_argument('--img', dest='img', default=None, help='待识别图片路径')
    parser.add_argument('--onnx', dest='onnx', default='./sbkuan.onnx', help='ONNX 模型路径')
    parser.add_argument('--show', dest='show', action='store_true', help='是否弹窗显示结果')
    parser.add_argument('--out', dest='out', default='./1_.png', help='输出图片路径')
    parser.add_argument('--ep', dest='ep', default='auto', help='执行提供器：auto/cpu/cuda/coreml，多项用逗号分隔')
    args = parser.parse_args()

    imgpa = args.img
    if not imgpa:
        for p in ['images/1.jpg', 'images/1.png', 'images/2.jpg', 'images/2.png']:
            if os.path.exists(p):
                imgpa = p
                break
    if not imgpa or not os.path.exists(imgpa):
        raise FileNotFoundError('未找到可用的图片，请使用 --img 指定，或放置到 images/ 下。')

    # 解析 EP 参数
    ep_map = {
        'cpu': ['CPUExecutionProvider'],
        'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
    }
    providers = None
    if args.ep and args.ep.lower() != 'auto':
        # 支持逗号分隔的 provider 名（如: cuda,cpu）或简写
        items = [x.strip() for x in args.ep.split(',') if x.strip()]
        expanded: list[str] = []
        for it in items:
            expanded.extend(ep_map.get(it.lower(), [it]))
        providers = expanded

    s = getTpInfo(args.onnx, providers=providers)
    s.shibie(imgpa, show=args.show, save_path=args.out)
