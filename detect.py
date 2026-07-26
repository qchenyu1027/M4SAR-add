# import cv2
# import torch
# import numpy as np
# import os
# from ultralytics import YOLO
# from ultralytics.utils.ops import non_max_suppression

# # ==========================================
# # 1. 基础配置与颜色映射
# # ==========================================
# class_names = [
#     'bridge', 'harbor', 'oil_tank',
#     'playground', 'airport', 'wind_turbine'
# ]

# colors = [
#     (255, 0, 0),   # 蓝色: bridge
#     (0, 255, 0),   # 绿色: harbor
#     (0, 0, 255),   # 红色: oil_tank
#     (255, 255, 0), # 青色: playground
#     (255, 0, 255), # 品红: airport
#     (0, 255, 255)  # 黄色: wind_turbine
# ]

# # 路径设置 (请核对)
# optical_dir = '/home/data1/zhb/dataset/M4-SAR/optical/images/test/'
# sar_dir = '/home/data1/zhb/dataset/M4-SAR/sar/images/test/'
# weights_path = 'runs/train/ICAFusion/yolo11-obb-ICAFusion-300e/weights/best.pt'

# out_vis_opt = 'runs/predict/M4-SAR/yolo11-obb-ICAFusion-300e/custom_vis/optical'
# out_vis_sar = 'runs/predict/M4-SAR/yolo11-obb-ICAFusion-300e/custom_vis/sar'
# os.makedirs(out_vis_opt, exist_ok=True)
# os.makedirs(out_vis_sar, exist_ok=True)

# CONF_THRES = 0.3 
# IOU_THRES = 0.45

# def main():
#     # ==========================================
#     # 2. 加载底层模型 (绕过易报错的高层接口)
#     # ==========================================
#     print(f"Loading dual-stream model from {weights_path}...")
#     full_model = YOLO(weights_path)
#     model = full_model.model  # 直接提取底层 PyTorch 模型
#     model.eval()              # 设置为推理模式
#     device = next(model.parameters()).device

#     img_names = [f for f in os.listdir(optical_dir) if f.endswith(('.jpg', '.png'))]
#     print(f"Found {len(img_names)} image pairs. Starting robust inference...")

#     # ==========================================
#     # 3. 逐图推理与画框
#     # ==========================================
#     for img_name in img_names:
#         opt_path = os.path.join(optical_dir, img_name)
#         sar_path = os.path.join(sar_dir, img_name)

#         if not os.path.exists(sar_path):
#             continue

#         # 读取图片
#         img_opt_raw = cv2.imread(opt_path)
#         img_sar_raw = cv2.imread(sar_path)
        
#         img_opt = cv2.resize(img_opt_raw, (512, 512))
#         img_sar = cv2.resize(img_sar_raw, (512, 512))

#         disp_opt = img_opt.copy()
#         disp_sar = img_sar.copy()

#         # 解决 negative strides 报错：使用 .copy()
#         tensor_opt = torch.from_numpy(img_opt[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
#         tensor_sar = torch.from_numpy(img_sar[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0

#         # 构建 6 通道输入张量 [1, 6, 512, 512]
#         input_tensor = torch.cat([tensor_opt, tensor_sar], dim=0).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # 纯底层推理：绝不吞通道
#             preds = model(input_tensor)
            
#             # 使用 YOLO 官方函数进行非极大值抑制 (NMS)，提取 OBB (旋转框)
#             # nc=6 表示有6个类别
#             results = non_max_suppression(preds, conf_thres=CONF_THRES, iou_thres=IOU_THRES, nc=6)
            
#             det = results[0] # 取第一张图的结果
            
#         # ==========================================
#         # 4. 解析 OBB 结果并画图
#         # ==========================================
#         if len(det):
#             # 旋转框格式转换解析: det 包含 [x1, y1, x2, y2, x3, y3, x4, y4, conf, cls] 或极坐标形式
#             # 兼容 Ultralytics 的 OBB 输出
#             try:
#                 # YOLOv8/v11 OBB 结构: det 是一个形状为 (N, 10) 或 (N, 7) 的 tensor
#                 # 如果是 (N, 7): [x_center, y_center, width, height, angle, conf, cls]
#                 # 我们这里借助 full_model 自带的解码机制处理最安全
#                 wrapper_results = full_model.postprocess(preds, input_tensor, [disp_opt.shape[:2]])
#                 res = wrapper_results[0]
                
#                 if res.obb is not None:
#                     obbs = res.obb.xyxyxyxy.cpu().numpy()
#                     cls_indices = res.obb.cls.cpu().numpy().astype(int)
#                     confs = res.obb.conf.cpu().numpy()
                    
#                     for obb, cls_idx, conf in zip(obbs, cls_indices, confs):
#                         if conf < CONF_THRES or cls_idx < 0 or cls_idx >= len(class_names):
#                             continue
                        
#                         color = colors[cls_idx]
#                         points = [(int(np.clip(round(pt[0]), 0, 511)), int(np.clip(round(pt[1]), 0, 511))) for pt in obb]
#                         pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        
#                         cv2.polylines(disp_opt, [pts], isClosed=True, color=color, thickness=2)
#                         cv2.polylines(disp_sar, [pts], isClosed=True, color=color, thickness=2)
                        
#             except Exception as e:
#                 print(f"[{img_name}] OBB解析跳过，原因: {e}")

#         # 保存结果
#         cv2.imwrite(os.path.join(out_vis_opt, img_name), disp_opt)
#         cv2.imwrite(os.path.join(out_vis_sar, img_name), disp_sar)

#     print(f"\n✅ 全部可视化完成！")
#     print(f"光学结果: {out_vis_opt}")
#     print(f"SAR结果: {out_vis_sar}")

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import os

# ==========================================
# 1. 基础配置与路径设置
# ==========================================
# 图像路径 (保持与原代码一致的测试集路径)
optical_img_dir = '/home/data1/zhb/dataset/M4-SAR/optical/images/test/'
sar_img_dir = '/home/data1/zhb/dataset/M4-SAR/sar/images/test/'

# 真实标签路径 (你提供的路径)
optical_label_dir = '/home/data1/zhb/dataset/M4-SAR/optical/labels/test'
sar_label_dir = '/home/data1/zhb/dataset/M4-SAR/sar/labels/test'

# 输出可视化路径
out_vis_opt = 'runs/gt_vis/optical'
out_vis_sar = 'runs/gt_vis/sar'
os.makedirs(out_vis_opt, exist_ok=True)
os.makedirs(out_vis_sar, exist_ok=True)

# 统一设定为红色框线 (OpenCV 中颜色通道为 BGR，所以红色是 0, 0, 255)
BOX_COLOR = (0, 0, 255) 
THICKNESS = 2

def draw_gt_boxes(img_path, label_path, save_path):
    """
    读取单张图像及其对应的 txt 标签，绘制红色真实框并保存
    """
    if not os.path.exists(img_path):
        return

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: 无法读取图像 {img_path}")
        return

    h, w = img.shape[:2]

    # 如果标签文件存在，则读取并画框 (不存在说明是背景图/无目标)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            # YOLO OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
            if len(parts) >= 9:
                # 提取 8 个坐标点并转换为 float
                coords = np.array(parts[1:9], dtype=np.float32)
                
                # 反归一化：将 0~1 的坐标还原为图像上的实际像素坐标
                coords[0::2] *= w  # x1, x2, x3, x4 乘以图像宽度
                coords[1::2] *= h  # y1, y2, y3, y4 乘以图像高度
                
                # 组装为 4个顶点的坐标矩阵
                pts = coords.reshape((4, 2)).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # 绘制闭合的多边形旋转框
                cv2.polylines(img, [pts], isClosed=True, color=BOX_COLOR, thickness=THICKNESS)

    # 保存绘制好真实框的图像
    cv2.imwrite(save_path, img)


def main():
    print("开始绘制真实标签框 (Ground Truth)...")
    
    # 获取光学图像列表
    img_names = [f for f in os.listdir(optical_img_dir) if f.endswith(('.jpg', '.png'))]
    print(f"共找到 {len(img_names)} 张测试图像，准备绘制红色真实框。")

    for img_name in img_names:
        # 1. 构造文件名 (标签文件通常与图像同名，后缀为 .txt)
        label_name = os.path.splitext(img_name)[0] + '.txt'

        # 2. 构造光学流的路径
        opt_img_path = os.path.join(optical_img_dir, img_name)
        opt_label_path = os.path.join(optical_label_dir, label_name)
        opt_save_path = os.path.join(out_vis_opt, img_name)
        
        # 3. 构造 SAR 流的路径
        sar_img_path = os.path.join(sar_img_dir, img_name)
        sar_label_path = os.path.join(sar_label_dir, label_name)
        sar_save_path = os.path.join(out_vis_sar, img_name)

        # 4. 分别执行绘制
        draw_gt_boxes(opt_img_path, opt_label_path, opt_save_path)
        draw_gt_boxes(sar_img_path, sar_label_path, sar_save_path)

    print(f"\n✅ 全部真实框可视化完成！")
    print(f"光学 GT 结果保存在: {out_vis_opt}")
    print(f"SAR GT 结果保存在: {out_vis_sar}")

if __name__ == "__main__":
    main()