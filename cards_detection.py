import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def find_features(img1):
    """Функцияя для нахождения признаков карт"""
    correct_matches_dct = {}
    directory = '/Users/rpdg/Downloads/sample_cards/'
    example_matches = []
    for image in os.listdir(directory):
        img2 = cv2.imread(directory + image, 0)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        # final_img = cv2.drawMatches(img1, kp1,
        #                             img2, kp2, matches[:20], None)
        #
        # final_img = cv2.resize(final_img, (1000, 650))
        # cv2.imshow("Matches", final_img)
        # cv2.waitKey()
        # matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:10], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        example_matches.append(img3)


        correct_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                correct_matches.append([m])
                correct_matches_dct[image.split('.')[0]] = len(correct_matches)

    correct_matches_dct = dict(sorted(correct_matches_dct.items(),
                                      key=lambda item: item[1], reverse=True))

    # отображение совпадений
    plt.figure(figsize=[20, 5])
    plt.subplot(151), plt.imshow(example_matches[0])
    plt.subplot(152), plt.imshow(example_matches[1])
    plt.subplot(153), plt.imshow(example_matches[2])
    plt.subplot(154), plt.imshow(example_matches[3])
    plt.subplot(155), plt.imshow(example_matches[4])
    plt.show()

    return list(correct_matches_dct.keys())[0]


def find_contours_of_cards(image):
    """Функция для нахождения контура изображения"""
    thresh = 170
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    _, thresh_img = cv2.threshold(blurred, thresh, 255,
                                  cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh_img,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def find_coordinates_of_cards(cnts, image):
    """Функция для нахождения координат карт"""
    cards_coordinates = {}
    for i in range(0, len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])
        if w > 20 and h > 30:
            img_crop = image[y - 15:y + h + 15,
                       x - 15:x + w + 15]
            cards_name = find_features(img_crop)
            cards_coordinates[cards_name] = (x - 15,
                                             y - 15, x + w + 15, y + h + 15)
    return cards_coordinates


def draw_rectangle_aroud_cards(cards_coordinates, image):
    """Функция для выделения карт"""
    for key, value in cards_coordinates.items():
        rec = cv2.rectangle(image, (value[0], value[1]),
                            (value[2], value[3]),
                            (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (36, 255, 12), 1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    my_image = cv2.imread('/Users/rpdg/Downloads/photo_2023-07-02 14.40.58.jpeg')
    gray_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
    contours = find_contours_of_cards(gray_image)
    cards_location = find_coordinates_of_cards(contours, gray_image)
    draw_rectangle_aroud_cards(cards_location, my_image)

    img_contours = np.zeros(my_image.shape)
    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
    cv2.imshow('contours', img_contours)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
