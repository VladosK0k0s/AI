import cv2
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from abc import ABC, abstractmethod


def create_ORB_descriptor(nfeatures=1000):
    alg = cv2.ORB_create(nfeatures=nfeatures)
    size = 32
    name = "ORB"
    return alg, size, name, nfeatures


def create_AKAZE_descriptor(nfeatures=500):
    alg = cv2.AKAZE_create()
    size = 61
    name = "AKAZE"
    return alg, size, name, nfeatures


def create_BRISK_descriptor(nfeatures=500):
    alg = cv2.BRISK_create(thresh=100)
    size = 64
    name = "BRISK"
    return alg, size, name, nfeatures


def predict(frame, model, descriptor):
    try:
        k, d = descriptor.alg.detectAndCompute(frame, None)
        dest_matches = np.zeros(
            (descriptor.nfeatures, descriptor.size)
        )
        for i in range(min(len(d), len(dest_matches))):
            dest_matches[i, :] = d[i, :]
        pure_data = dest_matches.ravel() / 256
        return model.predict(np.expand_dims(pure_data, axis=0))
    except:
        return 0


def process_video(model, descriptor, video_path, output_size, fps=30):
    cap = cv2.VideoCapture(video_path)
    images = []
    fourcc = cv2.VideoWriter_fourcc(*"FMP4")
    out = cv2.VideoWriter(
        "output/" + descriptor.name + "_out.avi", fourcc, fps, output_size
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if predict(frame, model, descriptor):
                text = "Object has Found"
            else:
                text = "Object wasn't Found"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            output_img = cv2.resize(frame, output_size)
            out.write(output_img)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def fit(classifier, descriptor, dir_that_contain_object, dir_that_not_contain_object):
    is_filled_model_exists = isfile('filled_model.pickle')
    if(is_filled_model_exists):
        print('filled_model exists')
        with open("filled_model.pickle", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        print('filled_model NOT exists')
        train_data = []
        y = []
        for i, dir_name in enumerate([dir_that_not_contain_object, dir_that_contain_object]):
            files_list = listdir(dir_name)
            for file_name in files_list:
                img = cv2.imread(dir_name + "/" + file_name, 0)
                k, d = descriptor.alg.detectAndCompute(img, None)
                try:
                    dest_matches = np.zeros(
                        (descriptor.nfeatures, descriptor.size))
                    for j in range(min(len(d), len(dest_matches))):
                        dest_matches[j, :] = d[j, :]
                    train_data.append(dest_matches.ravel() / 256)
                    y.append(i)
                except:
                    print(f"frame detecting error, continuing")
        train_data = np.array(train_data)
        y = np.array(y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            train_data, y, random_state=0, test_size=0.5
        )
        classifier.fit(X_train, Y_train)
        with open("filled_model.pickle", "wb") as f:
            pickle.dump(classifier, f)
        return classifier


class Descriptor:
    def __init__(self, decriptor_alg, decriptor_size, decriptor_name, decriptor_nfeatures):
        self.alg = decriptor_alg
        self.size = decriptor_size
        self.nfeatures = decriptor_nfeatures
        self.name = decriptor_name


def start_recording(model, descriptor):
    cap = cv2.VideoCapture(0)
    replaced_img = cv2.imread("img.jpg")
    override_image = cv2.imread("photo.jpg")
    height, width, _ = replaced_img.shape
    override_image = cv2.resize(override_image, (width, height))

    k, d = descriptor.alg.detectAndCompute(replaced_img, None)

    bf = cv2.BFMatcher()
    while True:
        success, imgWebcam = cap.read()
        imgAug = imgWebcam.copy()
        k2, d2 = descriptor.alg.detectAndCompute(imgAug, None)
        try:
            matches = bf.knnMatch(d, d2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            data = predict(imgWebcam, model, descriptor)
            if predict(imgWebcam, model, descriptor):
                srcPts = np.float32(
                    [k[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dstPts = np.float32(
                    [k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(
                    srcPts, dstPts, cv2.RANSAC, 5)
                pts = np.float32(
                    [[0, 0], [0, height], [width, height], [width, 0]]
                ).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                imgWarp = cv2.warpPerspective(
                    override_image, matrix, (imgWebcam.shape[1],
                                             imgWebcam.shape[0])
                )

                maskNew = np.zeros(
                    (imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
                cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                maskInv = cv2.bitwise_not(maskNew)
                imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
                imgAug = cv2.bitwise_or(imgWarp, imgAug)
        except:
            # print("Bad frame")
            pass

        # out.write(imgAug)
        cv2.imshow("AugmentedReality", imgAug)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    classifier = MultinomialNB()
    decriptor_alg, decriptor_size, decriptor_name, decriptor_nfeatures = create_BRISK_descriptor()
    descriptor = Descriptor(decriptor_alg, decriptor_size,
                            decriptor_name, decriptor_nfeatures)
    model = fit(classifier, descriptor, "contain", "not_contain")

    # process_video(model, descriptor, "data/test.mp4", (640, 480), fps=30)

    start_recording(model, descriptor)


main()
