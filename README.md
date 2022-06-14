# for_smart_traffic
This is a fraction of our project where we attempted to build a smart control for the traffic light. We basically started with a static controlled taffic signals as the customer wanted that particular specificaion. We designed the casing of traffic light, developed it in our custom made lab. As we had simple traffic models with fixed time control, we tried to put a smart touch in it. Basic object detecting sensors, viz. IR sensors with transmitter and receiver were implemented with AT-MEGA as control chip. However we could not make the system practically feasible even in our lab environment.
So, we opted to bring Artifical Intelligence into our system. Optimizing traffic control and pushing the static control to dynamic control is clearly the demand today. We just wanted to propose a demo model of a dynamic control. Our system utilizes images from cameras at traffic junctions for traffic density calculation using image processing and AI. It focuses on the algorithm for switching traffic lights based on the density to reduce congestion, thereby providing faster transit and making a progrssive impact in the existing scenario.

However, we could not achieve traffic video monitoring and surveillance systems's data, so we opted to make our own video recordings and custom labelling the images. We tried this but the loss of our system could not be maintained at a desirable value.  So, we thought just to make a demo model. Then we implemented the COCO dataset and developed code that just identified the vehicle class we wanted. We custmozied YOLO algorithm as per our needs and trained our model. The training loss was at a level that our control system could digest. The validation loss was also at a convining level. Then we tested our system with the images from our raods, the detections were good, however if only we could train the model with our own own dataset, the detections would be much better. And finally the the traffic lights control was as we expected.

This repo contains just the AI nad image processing segment of our project.

# busy_asan


https://user-images.githubusercontent.com/78263671/173554545-a50fbe4d-0971-4a1d-b50b-f8c16fe78a52.mp4

