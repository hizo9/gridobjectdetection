# Imports
import cv2
import time
from ultralytics import YOLO

# Configs
model = YOLO("yolo11n.pt")

fps = 0
frame_count = 0
start_time = time.time()

show_class_names = False 

# DO NOT TOUCH
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, show=False, conf=0.25) # conf is confidence threshold

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time

    detected_grids = set()  # keep track of which grids have detections
    height, width, _ = frame.shape

    for result in results:
        boxes = result.boxes
        classes = result.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
            conf = box.conf[0]  # get confidence
            cls = int(box.cls[0])  # get class index
            label = f"{classes[cls]}: {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # class name toggle
            if show_class_names:
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # detect which grid the object is in
            grid_x = int((x1 + x2) / 2) // (width // 3)
            grid_y = int((y1 + y2) / 2) // (height // 3)
            detected_grids.add((grid_x, grid_y))

    # display FPS on frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # draw 3x3 grid
    grid_color = (255, 255, 255)  # White color for the grid
    grid_thickness = 1

    # vertical lines
    for i in range(1, 3):
        x = int(i * width / 3)
        cv2.line(frame, (x, 0), (x, height), grid_color, grid_thickness)

    # horizontal lines
    for i in range(1, 3):
        y = int(i * height / 3)
        cv2.line(frame, (0, y), (width, y), grid_color, grid_thickness)

    # check if grids >= 5
    if len(detected_grids) >= 5:
        cv2.putText(frame, "Barrier almost full", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # display number of filled grids
    cv2.putText(frame, f'Filled Grids: {len(detected_grids)}', (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # show results
    cv2.imshow("YOLO Results", frame)

    # break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()