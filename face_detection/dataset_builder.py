"""Class to build the dataset of face images
"""

import os
import cv2
from face_detection.operations import detect_face, cut_faces

class DatasetBuilder(object):
    """ Class to build image dataset

    This class contains the modules necessary to build a the dataset
    of faces images.
    """

    def __init__(self, root, dst, num=1):
        self.origin = root
        self.destination = dst
        self.max_faces = num

    def get_images_path(self, folders,
                        extensions=['jpg', 'jpeg', 'png'],
                        max_num_imgs=5000):
        """Module to find the full path to images with desired
        extensions.

        :param folders: Folders names to look for images
        :param extensions: Extensions of images to find
        :param max: Maximum number of images to check
        :type folders: List of Strings
        :type extensions: List of Strings
        :type max: Integer
        :returns: All the paths of the images
        :rtype: List of strings
        """
        print max_num_imgs
        images_path = []
        for folder in folders:
            folder_path = self.origin + folder
            print 'Getting Images from ' + folder_path
            for path, _, filenames in os.walk(folder_path):
                print filenames
                for filename in filenames:
                    if filename.split(".")[-1].lower() in extensions:
                        images_path.append(os.path.join(path, filename))
                    if len(images_path) == max_num_imgs:
                        break
        print '\nImages retrieved: ' + str(len(images_path))
        return images_path

    def get_faces(self, images_path):
        """Module to obtain the faces from the images

        This module will try to find a face in each of the images that
        correspond fo the input parameter list with paths. If a face is
        found it crops the face from the image and adds it to the list
        of found faces. THen returns this list of found faces together
        with the filename.

        :param images_path: Path of all images to check for a faces_coord
        :type images_path: List of Strings
        :returns: images of faces found and the filename corressponding
        to each image.
        strings with the desired filenames.
        :rtype: tuple of lists. First list contain numpy arrays. Second
        contains strings.
        """

        print 'Number of Images to analyze: ' +  str(len(images_path))
        count = 0
        faces_image = []
        faces_filename = []
        for i, image_path in enumerate(images_path):
            print '\nAnalyzing Image: ' + str(i+1) + '/' + \
                   str(len(images_path))
            print 'Path: ' + image_path

            image = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_coords = detect_face(image_gray, self.max_faces)
            if len(face_coords) == 0:
                for i in xrange(3):
                    (h, w) = image_gray.shape
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, 90, 1)
                    image = cv2.warpAffine(image, M, (h, w))
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face_coords = detect_face(image_gray, self.max_faces)
                    if len(face_coords) != 0:
                        break
            if len(face_coords) != 0:
                print "Face found"
                faces_image.extend(cut_faces(image, face_coords))
                faces_filename.append(image_path.split("/")[-1])
                prefix = 2
                if len(face_coords) > 1:
                    faces_filename.append(str(prefix) + "_" +
                                          image_path.split("/")[-1])
                    prefix += 1
                count += 1
            else:
                print "Face not found"
        return (faces_image, faces_filename)

    def save_images(self, images, filenames):
        """ This modules saves images in the destination folder

        Each image is save in the destination folder with the
        corresponding filename from the parameters.

        :param images: list of image
        :param filenames: intented filenames for the images
        :type images: list of numpy arrays
        :type filenames: list of strings
        :returns: Nothing
        """
        print '\nSaving images...'
        for i, filename in enumerate(filenames):
            full_filename = os.path.join(self.destination, filename)
            cv2.imwrite(full_filename, images[i])
            print 'Image saved: ' + full_filename
