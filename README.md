
# YOLOv8- Vehicle Detection and Counting

#### Description:
This project utilizes YOLOv8 for vehicle detection and ByteTrack for tracking. YOLOv8 detects vehicles in video footage, while ByteTrack tracks their movement across frames. Vehicles crossing a predefined line (LineZone) are counted, and the results, including vehicle counts and annotations, are saved in an output video file.

### Steps to run the Script

#### Clone the repository:
``` bash 
git clone https://github.com/your-username/yolo-vehicle-detection.git

!cd yolo-vehicle-detection

```
#### Install the required Python packages:

``` bash

pip install -r requirements.txt

```
Download the YOLOv8 (best.pt) weights from below link .







#### Weights

[link](https://drive.google.com/file/d/1kLpQeYHJGMbEORxZI0iNcFVAxmUWAKKi/view?usp=sharing)


#### Run the script with the following command:

``` bash

python vehicle_detection.py --model "path/to/best.pt" --video_input "path/to/video.mp4" --output "path/to/output.mp4" --frame_interval 2
```

--model: Path to the trained YOLO model (e.g., best.pt).

--video_input: Path to the input video file.

--output: Path where the output video will be saved with annotations.

--frame_interval: Optional frame interval to process (default is 2).
