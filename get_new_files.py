import os
import os.path
import re
import glob
from shutil import copy



if __name__ == '__main__':

    large_image_path = r"C:\Users\lisak\NG\segmentation\hand_bigger\data\images\6"
    test_image_path = r"C:\Users\lisak\NG\segmentation\finger\data\images\8"
    val_image_path = r"C:\Users\lisak\NG\segmentation\finger\data\validation_images"


    directory_lim = os.fsencode(large_image_path)
    #
    # dir_src = r"/path/to/folder/with/subfolders"
    # dir_dst = r"/path/to/destination"

    t1 = "0e2b25fc-ef1de83b-Point_9891.jpg"
    filename_test1 = os.path.join(test_image_path, t1)
    statinfo = os.stat(filename_test1)
    print(statinfo)
    path_exists = os.path.exists(filename_test1.strip())
    print(path_exists)


    for file in glob.iglob('%s/**/*.jpg' % large_image_path, recursive=True):
        head, tail = os.path.split(file)
        filename_test = os.path.join(test_image_path, tail)
        # statinfo = os.stat(filename_test.strip())
        # print(statinfo)
        path_exists = os.path.exists(filename_test.strip())
        # if not os.path.exists(filename_test.strip()):
        if not path_exists:
            copy(file, val_image_path)
        else:
            print("file %s already exists" % filename_test)


    # for file in os.listdir(directory_lim):
    #     filename = os.fsdecode(file)
    #     filename_test = test_image_path + "/" + filename
    #     if not os.path.isfile(filename_test):
    #         save(val_image_path + "/" + filename)