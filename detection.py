
from imageai.Detection.Custom import CustomObjectDetection
import os
from collections import Counter


img_no="022"
execution_path = os.getcwd()
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("models/detection_model-"+img_no+".h5") 
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "photos/parkinglot/"+img_no+".png"), output_image_path=os.path.join(execution_path , "photos/parkinglot/result/result"+img_no+".png"))
number=0
space=0
for eachObject in detections:

    if eachObject["name"]=="car":
        number+=1
    if eachObject["name"]=="space":
        space+=1

print("駐車台数:",number,"  空きスペース：",space)
