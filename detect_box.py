import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

# ==========================================
# 1. 基础配置与颜色映射 (适配你的 M4-SAR 6分类)
# ==========================================
class_names = [
    'bridge', 'harbor', 'oil_tank',
    'playground', 'airport', 'wind_turbine'
]

colors = [
    (255, 0, 0),   # 蓝色: bridge
    (0, 255, 0),   # 绿色: harbor
    (0, 0, 255),   # 红色: oil_tank
    (255, 255, 0), # 青色: playground
    (255, 0, 255), # 品红: airport
    (0, 255, 255)  # 黄色: wind_turbine
]

optical_dir = '/home/data1/zhb/dataset/M4-SAR/optical/images/test/'
sar_dir = '/home/data1/zhb/dataset/M4-SAR/sar/images/test'
weights_path = 'runs/train/ICAFusion/yolo11-obb-ICAFusion-300e/weights/best.pt'

out_vis_opt = 'runs/predict/yolo11-obb-ICAFusion-300e/optical'
out_vis_sar = 'runs/predict/yolo11-obb-ICAFusion-300e/sar'
os.makedirs(out_vis_opt, exist_ok=True)
os.makedirs(out_vis_sar, exist_ok=True)

CONF_THRES = 0.3 
IOU_THRES = 0.45

def xywhr2xyxyxyxy(xywhr):
    """
    纯底层 PyTorch 实现：将网络输出的 [中心x, 中心y, 宽, 高, 旋转角] 
    转换为旋转包围框的 4 个顶点绝对坐标 [x1, y1], [x2, y2], [x3, y3], [x4, y4]
    """
    ctr = xywhr[:, :2]
    w, h, angle = xywhr[:, 2], xywhr[:, 3], xywhr[:, 4]
    cos_value, sin_value = torch.cos(angle), torch.sin(angle)
    
    vec1 = torch.stack([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.stack([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
    
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.stack([pt1, pt2, pt3, pt4], dim=1)

def main():
    # ==========================================
    # 2. 提取最纯粹的底层 PyTorch 模型
    # ==========================================
    print(f"Loading dual-stream model from {weights_path}...")
    full_model = YOLO(weights_path)
    model = full_model.model  
    model.eval()              
    device = next(model.parameters()).device

    img_names = [f for f in os.listdir(optical_dir) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(img_names)} image pairs. Starting robust inference...")

    # ==========================================
    # 3. 逐图推理与解码
    # ==========================================
    for img_name in img_names:
        opt_path = os.path.join(optical_dir, img_name)
        sar_path = os.path.join(sar_dir, img_name)

        if not os.path.exists(sar_path):
            continue

        img_opt_raw = cv2.imread(opt_path)
        img_sar_raw = cv2.imread(sar_path)
        
        img_opt = cv2.resize(img_opt_raw, (512, 512))
        img_sar = cv2.resize(img_sar_raw, (512, 512))

        disp_opt = img_opt.copy()
        disp_sar = img_sar.copy()

        # 解决 Negative Strides 内存断层问题
        tensor_opt = torch.from_numpy(img_opt[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
        tensor_sar = torch.from_numpy(img_sar[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0

        # 手动拼接 6 通道
        input_tensor = torch.cat([tensor_opt, tensor_sar], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)
            
            # 使用官方 NMS，nc=6 保障角度维度 (angle) 不会丢失
            results = non_max_suppression(preds, conf_thres=CONF_THRES, iou_thres=IOU_THRES, nc=6)
            det = results[0] 
            
        # ==========================================
        # 4. 手动解析旋转框矩阵并绘制 (彻底告别版本报错)
        # ==========================================
        if len(det):
            # 此时 det 的形状是 (N, 7): [x1, y1, x2, y2, conf, cls, angle]
            # 因为 NMS 内部会将中心点转换为外接矩形的对角坐标，我们需要先还原出网络预测的原始 w 和 h
            xyxy = det[:, :4]
            conf = det[:, 4]
            cls_indices = det[:, 5].int()
            angle = det[:, 6]
            
            x_c = (xyxy[:, 0] + xyxy[:, 2]) / 2
            y_c = (xyxy[:, 1] + xyxy[:, 3]) / 2
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            
            # 重新拼装为带旋转角的中心点格式：[N, 5]
            xywhr = torch.stack([x_c, y_c, w, h, angle], dim=-1)
            
            # 调用我们自己写的数学转换函数，直接获取 4 个顶点坐标
            obbs = xywhr2xyxyxyxy(xywhr).cpu().numpy() 
            cls_indices = cls_indices.cpu().numpy()
            confs = conf.cpu().numpy()
            
            for obb, cls_idx, cnf in zip(obbs, cls_indices, confs):
                if cnf < CONF_THRES or cls_idx < 0 or cls_idx >= len(class_names):
                    continue
                
                color = colors[cls_idx]
                
                # 限幅并转换为整数像素点
                points = [(int(np.clip(round(pt[0]), 0, 511)), int(np.clip(round(pt[1]), 0, 511))) for pt in obb]
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                
                # 双流画框
                cv2.polylines(disp_opt, [pts], isClosed=True, color=color, thickness=2)
                cv2.polylines(disp_sar, [pts], isClosed=True, color=color, thickness=2)

        cv2.imwrite(os.path.join(out_vis_opt, img_name), disp_opt)
        cv2.imwrite(os.path.join(out_vis_sar, img_name), disp_sar)

    print(f"\n✅ 全部可视化完成！")
    print(f"光学结果: {out_vis_opt}")
    print(f"SAR结果: {out_vis_sar}")

if __name__ == "__main__":
    main()