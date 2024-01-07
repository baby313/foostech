import cv2
import time

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 120)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

t0 = t1 = 0
fcount = 0
fps = 0
while True:
    ret, frame = cam.read()
    if ret:
        fcount += 1
        if fcount % 100 == 0:
            t1 = time.time()
            fps = 100/(t1-t0)
            t0 = t1
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()
