import cv2
import logging
import os
import glob
import multiprocessing as mp

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
logger = logging.getLogger(__name__)

def preprocess_image(input_path, output_path, crop_dim=160):

    img = cv2.imread(input_path)
    cv2.imshow("ben", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
    max = 0
    biggest_face = []
    for x, y, w, h in faces:
        if (w * h) > max:
            max = w * h
            biggest_face = [x, y, x + w, y + h]
    if (len(biggest_face) > 0):
        # img = cv2.rectangle(img, (biggest_face[0], biggest_face[1]), (biggest_face[2], biggest_face[3]), (0, 255, 0), 3)
        crop_img = img[biggest_face[1]:biggest_face[3],biggest_face[0]:biggest_face[2]]
        resized = cv2.resize(crop_img,(crop_dim, crop_dim))
        cv2.imwrite(output_path, resized)

        # cv2.imshow("ben", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        logger.info("No face detected.")

if __name__ == '__main__':
    image_dir = 'data'
    logging.basicConfig(level=logging.INFO)
    # preprocess_image("mountain.jpg")
    if not os.path.exists("preprocessed_data"):
        os.makedirs("preprocessed_data")
    pool = mp.Pool(processes=mp.cpu_count())
    for image_dir in os.listdir(image_dir):
        image_output_dir = os.path.join("preprocessed_data", os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(image_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join("preprocessed_data", os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        pool.apply_async(preprocess_image, (image_path, output_path, 160))

    pool.close()
    pool.join()

