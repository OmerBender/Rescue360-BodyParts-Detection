import cv2
import time

for idx in [0, 1]:
    print(f"Trying camera {idx} for 10 seconds...")

    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print(f"Camera {idx}: not opened")
        continue

    start = time.time()

    while time.time() - start < 10:
        ret, frame = cap.read()

        if not ret:
            print(f"Camera {idx}: no frame")
            continue

        cv2.putText(
            frame,
            f"CAMERA INDEX {idx}",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            4,
        )

        cv2.imshow(f"Camera {idx}", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(1)

print("Done.")
