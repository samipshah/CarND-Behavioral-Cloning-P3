import csv
import numpy as np
import cv2
import sklearn
import platform
from sklearn.model_selection import train_test_split
from random import shuffle
import random
import matplotlib.pyplot as plt
import gc

if platform.system() == 'Darwin':
    MOUNT = "./data"
else:
    MOUNT = "/intput"


def get_samples():
    samples = []
    with open(MOUNT + "/driving_log.csv") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(
        samples, test_size=0.2)
    return train_samples, validation_samples

def steering_corrector():
    # 1 / 20 radians
    return 0.25

def random_shear(image, steering, shear_range, debug=False):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    if debug is True:
        print('dx: ', dx)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = 0.004*dx
    if debug is True:
        print('dsteering: ', dsteering)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering += dsteering

    return image, steering

def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4 * (2 * np.random.uniform() - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image, steering):
    coin = np.random.randint(0, 2)
    if coin == 0:
        image, steering = cv2.flip(image, 1), -steering
    return image, steering


def process_image_steering(img_path, steering, debug=False):
    image = cv2.imread(img_path)
    #image = augment_brightness_camera_images(image)
    if debug is True:
        print('steering :', steering)
        plt.figure()
        plt.imshow(image)

    # image, steering = random_shear(image, steering, shear_range=100, debug=debug)
    # if debug is True:
    #     print('steering :', steering)
    #     plt.figure()
    #     plt.imshow(image)

    image, steering = random_flip(image, steering)
    if debug is True:
        print('steering :', steering)
        plt.figure()
        plt.imshow(image)
        plt.show()

    image = random_brightness(image)
    if debug is True:
        print('steering :', steering)
        plt.figure()
        plt.imshow(image)
        plt.show()

    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
    if debug is True:
        print('steering :', steering)
        plt.figure()
        plt.imshow(image)
        plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    if debug is True:
        print('steering :', steering)
        plt.figure()
        plt.imshow(image)
        plt.show()
        
    return image, steering


STEERING_THRESHOLD = 0.1

def generator(samples, batch_size=32, threshold=0.2):
    num_samples = len(samples)
    print("Threshold: {.3f}", threshold)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                correction = steering_corrector()
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                name = MOUNT + '/IMG/' + batch_sample[0].split('/')[-1]
                name_left = MOUNT + '/IMG/' + batch_sample[1].split('/')[-1]
                name_right = MOUNT + '/IMG/' + batch_sample[2].split('/')[-1]
                center_image, steering_center = process_image_steering(
                    name, steering_center)
                left_image, steering_left = process_image_steering(
                    name_left, steering_left)
                right_image, steering_right = process_image_steering(
                    name_right, steering_right)

                if abs(steering_center) > STEERING_THRESHOLD or np.random.uniform() > threshold:
                    images.append(center_image)
                    angles.append(steering_center)

                if abs(steering_left) > STEERING_THRESHOLD or np.random.uniform() > threshold:
                    images.append(left_image)
                    angles.append(steering_left)

                if abs(steering_right) > STEERING_THRESHOLD or np.random.uniform() > threshold:
                    images.append(right_image)
                    angles.append(steering_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def data_visualization():
    training_set, validation_set = get_samples()
    angles = []
    for batch_sample in (training_set + validation_set):
        center_angle = float(batch_sample[3])
        angles.append(center_angle)
        # angles.append(-center_angle)

    # trim image to only see section with road
    # y_train = np.array(angles)
    # plt.hist(y_train, bins=20)
    # plt.ylabel('Steering Angle')
    #plt.show()

    # transform data randomly
    test_set = [training_set[0]]
    for batch_sample in (training_set + validation_set):
        if float(batch_sample[3]) > 0.0:
            test_set.append(batch_sample)
            break

    # image visualization
    for batch_sample in test_set:
        steering_center = float(batch_sample[3])
#        if steering_center > 0.35 or steering_center < -0.35:
#            continue
        correction = steering_corrector()
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        name = MOUNT + '/IMG/' + batch_sample[0].split('/')[-1]
        name_left = MOUNT + '/IMG/' + batch_sample[1].split('/')[-1]
        name_right = MOUNT + '/IMG/' + batch_sample[2].split('/')[-1]
        center_image, steering_center = process_image_steering(
            name, steering_center, debug=True)

    return

    images = []
    angles = []
    i = 0
    threshold = 0.2
    for batch_sample in (training_set + validation_set):
        steering_center = float(batch_sample[3])
        correction = steering_corrector()
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        name = MOUNT + '/IMG/' + batch_sample[0].split('/')[-1]
        name_left = MOUNT + '/IMG/' + batch_sample[1].split('/')[-1]
        name_right = MOUNT + '/IMG/' + batch_sample[2].split('/')[-1]
        if abs(steering_center) > STEERING_THRESHOLD or np.random.uniform() > threshold:
            center_image, steering_center = process_image_steering(
            name, steering_center, debug=False)
        if abs(steering_left) > STEERING_THRESHOLD or np.random.uniform() > threshold:
            left_image, steering_left = process_image_steering(
            name_left, steering_left, debug=False)
        if abs(steering_right) > STEERING_THRESHOLD or np.random.uniform() > threshold:
            right_image, steering_right = process_image_steering(
            name_right, steering_right, debug=False)

        angles.append(steering_center)
        angles.append(steering_left)
        angles.append(steering_right)
        i += 1
        if i % 500 == 0:
            gc.collect()

    y_train = np.array(angles)
    plt.hist(y_train, bins=20)
    plt.ylabel('Steering Angle')
    plt.show()


if __name__ == "__main__":
    data_visualization()
    # training, validation = get_samples()
    # train_generator = generator(training)
    # valid_generator = generator(validation)
    # for train_sample in train_generator:
    #     print(train_sample[0])
    #     print(train_sample[1])
