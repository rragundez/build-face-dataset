"""Functions for manipulations of images
"""

import cv2

# path to the trained models by opencv to detect frontal faces
path = 'face_detection/haarcascades/'
frontal = path + 'haarcascade_frontalface_alt.xml'

# Create classifier objects
face_class = cv2.CascadeClassifier(frontal)

def detect_face(image_gray, max_faces):
    """ Function to detect faces in an image.

    This function will use the OpenCV model to find faces in the image.
    It can find many faces but it will return just the first face.

    :param image_gray: image in grayscale to look for a face
    :param max_faces: maximum number of faces to detect in a single image
    :type image_gray: 2D numpy array
    :type max_faces: integer
    :returns: list containing the coordenates of a square surrounding
    the face or faces.
    :rtype: list of tuples of integers
    """
    # jump in size of sliding window. Recommended 1.2-1.3
    scale_factor = 1.2
    # Minimum numberof neighbor sliding windows find a face
    min_neighbors = 5
    # Sets the min_size of the face we want to detect
    min_size = (20, 20)
    # Defines how we want the algorithm to run
    flags = cv2.CASCADE_SCALE_IMAGE
    faces_coord = face_class.detectMultiScale(image_gray,
                                              scale_factor,
                                              min_neighbors,
                                              flags,
                                              min_size)
    return faces_coord[:max_faces]

def cut_faces(image, faces):
    """ Crop image rectangle around the detected face.

    This function can do two types of crop. By default tries to
    maximize the cropping area around the face, if this area turns
    turns to be to big, please uncomment the line inside the for. This
    will make the crop exactly as the output from the OpenCV detector.

    :param image: image where faces were detected
    :param faces_coord: coordenates for rectangle around faces
    :type image: numpy array
    :type faces_coord: list of tuples containing int coordenates
    :returns: images of faces found
    :rtype: list of numpyarrays
    """
    faces_image = []
    for (x, y, w, h) in faces:
        # people_faces.append(image[y: y + h, x: x + w])
        x_center = x + w / 2
        y_center = y + h / 2
        height, width = image.shape[:2]
        w_cut = min(x_center, width - x_center,
                    y_center, height - y_center)
        faces_image.append(image[y_center - w_cut : y_center + w_cut,
                                 x_center - w_cut : x_center + w_cut])
    return faces_image
