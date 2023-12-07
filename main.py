import cv2 as cv


def detect(video):
    while True:
        try:
            frame, img = video.read()

            plate_cascade = cv.CascadeClassifier("haarcascade_russian_plate_number.xml")
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, 1.5, 5)

            for (x, y, w, h) in plates:
                area = w * h

                if area > 500:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv.imshow("Video", img)

            # press esc to exit
            if cv.waitKey(10) and cv.waitKey(10) % 256 == 27:
                break

        except BaseException as e:
            print(e.args)
            break

    cv.destroyAllWindows()
    video.release()


video = cv.VideoCapture('./video.mp4')
detect(video)
