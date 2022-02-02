import cv2


def get_processed_image(state):
    image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    image = image[16:, 16:]
    image = image[:-16, :-16].astype(float)
    image /= 255.0
    # image = cv2.resize(image, (64, 64))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # print(image.shape)
    return image
