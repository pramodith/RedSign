Steps to run the code:


At the moment just the FRCNN is working well. After cloning the repo please go to the faster_rcnn_pytorch directory.
Use the saved model weights using the invite sent to you're gatech id's from pramodith@gatech.edu.

The downloaded weights should be placed in the following path:
faster_rcnn_pytorch/models/vgg16/pascal_voc.

Then run demo.py from faster_rcnn_pytorch using the command:
python demo.py --cuda.

If you face any dependency issues please download all the required packages mentioned in requirements.txt.
Pytorch 0.4 was used for the project.

On a side note all the results obtained on the test set can be viewed in the output_images directory. A confidence 
of 0.7 was set as the threshold for the bounding box to be considered a positive example.

You can train the YOLOv3 by using the command python DNN.py, initial training didn't seem promising so I didn't 
showcase the results on the test set for YOLOv3.

Please feel free to contact me if you run into any issues while running the code.
Credit should go to https://github.com/jwyang/faster-rcnn.pytorch and https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch for the faster_rcnn and YOLOv3 codebases that I used to suit my requirements.
