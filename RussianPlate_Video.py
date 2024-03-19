import time
import cv2

cap = cv2.VideoCapture('russian_plate.mp4')
plate_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')



while True:
    _,frame = cap.read()

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(frame_gray)


    for (x,y,w,h) in plates:
        roi_frame = frame[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi_frame,ksize=7)

        frame[y:y+h,x:x+w] = blurred_roi

    time.sleep(1/200)
    cv2.imshow('Russian Plate Blurring',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()