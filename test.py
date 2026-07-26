from ultralytics import YOLO
 
def main():
    model = YOLO('runs/train/agrkan/yolo11-obb-agrkan-300e/weights/best.pt')
    metrics = model.val(split='test', imgsz=512, device=0, batch=1, workers=4, project='runs/test/M4-SAR', name='yolo11-obb-IRD')
    map75 = metrics.box.map75
    print(map75)
  
if __name__ == '__main__':
    main()
