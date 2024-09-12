import numpy as np
import argparse
import supervision as sv
from ultralytics import YOLO
import cv2


def main(model_path, video_input, output_path, frame_interval=2):
    # Load the YOLO model
    model = YOLO(model_path)

    # Set up ByteTrack tracker and annotators
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.3, text_thickness=1, text_padding=0, border_radius=0)

    # Define LineZone to detect vehicles crossing a line
    start, end = sv.Point(x=0, y=250), sv.Point(x=1280, y=250)
    line_zone = sv.LineZone(start=start, end=end)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # Open video and set up video writer
    cap = cv2.VideoCapture(video_input)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))

    vehicle_counts = {name: 0 for name in model.names.values()}

    # Frame interval for processing (skip every 2nd frame to reduce load)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for performance optimization
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval != 0:
            continue

        # Make predictions and track the detections
        results = model.predict(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Update LineZone for detecting vehicles crossing the line
        crossed_in, crossed_out = line_zone.trigger(detections)

        # Assuming crossed_in and crossed_out are boolean arrays
        crossed_in_indices = np.where(np.array(crossed_in) == True)[0]
        crossed_out_indices = np.where(np.array(crossed_out) == True)[0]
        
        # Merge the two arrays
        merged_indices = np.concatenate((crossed_in_indices, crossed_out_indices))

        for i in merged_indices:
            label = detections.class_id[i]
            class_name = model.names[label]
            vehicle_counts[class_name] += 1

        # Annotate the frame with detection boxes, labels, and line crossing stats
        labels = [f"#{tracker_id} {model.names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]
        
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        line_annotator.annotate(annotated_frame, line_zone)

        # Display vehicle counts on the frame
        cv2.putText(annotated_frame, f"Total Vehicles: {sum(vehicle_counts.values())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset = 50
        for class_name, count in vehicle_counts.items():
            cv2.putText(annotated_frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 20

        # Show the frame with the annotations
        cv2.imshow("frame", annotated_frame)
        
        # Write the frame to the output video
        out.write(annotated_frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Vehicle Detection with LineZone Crossing")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLO model file (e.g., best.pt)")
    parser.add_argument("--video_input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video with annotations")
    parser.add_argument("--frame_interval", type=int, default=2, help="Frame interval for processing (default: 2)")

    args = parser.parse_args()
    
    main(args.model, args.video_input, args.output, args.frame_interval)
