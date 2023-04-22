#캠 정상작동 확인을 위한 코드
import cv2

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if ret:
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
