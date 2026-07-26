from ultralytics import YOLO
import  os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
    data ='ultralytics/cfg/datasets/M4-SAR.yaml'                                                                                                                                                                
    # cfg = 'ultralytics/cfg/models/benchmark/yolo11-obb-ICAFusion.yaml'
    # model = YOLO('runs/train/S2KANFUSION/yolo11-obb-S2KANFUSION/weights/last.pt')
    model = YOLO('runs/train/S2KANFUSION/yolo11-obb-S2KANFUSION-300e2/weights/best.pt')
    project = 'runs/train/S2KANFUSION/'
    name = 'yolo11-obb-S2KANFUSION-513'
    model.train(data=data, epochs=200, batch=64, imgsz=512, name=name, resume=False, device=0, project=project,patience=None,workers=8,cache=False)
    # Single-GPU training: device=0
    # Multi-GPU training: device=[0,1]'
    # resume train setting 'resume=True' and 'cfg = 'runs/train/M4-SAR/yolo11-obb-E2E-OSDet-300e/weights/last.pt''

if __name__ == '__main__':
    main()
