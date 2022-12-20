import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm.auto import tqdm
import re

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

circle_border_radius = 4
drawing_thickness = 10
DS = 4

arm_ratio = 5./26. # ratio between arm width and length
hand_ratio = 0.5/2. # Ration between finger segment  width and length

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# arm_connections = [[11,13],[13, 15], [12, 14], [14,16]]#, [15, 19, 17], [16, 18, 20]]
arm_connections = [[12, 14],[14,16]]#, [15, 19, 17], [16, 18, 20]]

hand_triangle_connections = [0, 5, 9, 13, 17]

def circle_segmentation(image, start, end, ratio = arm_ratio, color = WHITE_COLOR):

    if start is None  or end is None:
        return
    
    start, end = np.array(start), np.array(end)
    d = np.linalg.norm(np.array(start) - np.array(end))

    radius = int(d * ratio)
    S = np.linspace(0, 1, int(d / DS))
    for s in S:
        px = list(np.array(start) + s * (np.array(end)-np.array(start)))
        px[0], px[1] = int(px[0]), int(px[1])
        cv2.circle(image, px, radius, color, drawing_thickness)


def draw_hand(image, maskImage, landmark_list, connections, arm = False):
    image_rows, image_cols, _ = image.shape

    if not landmark_list or not connections:
        return

    PX = []
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        PX.append(landmark_px)

    # Draws the connections if the start and end landmarks are both visible.
    for i, connection in enumerate(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        if np.any(i == np.array([1,3])):
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio = hand_ratio*0.4)
        elif np.any(i == np.array([8, 5, 13])):
            continue
        elif np.any(i == np.array([10, 7, 11, 2])):
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio = hand_ratio*0.5)
        else:
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio = hand_ratio)
    pts = np.array([PX[i] for i in hand_triangle_connections], np.int32)
    cv2.fillPoly(maskImage, [pts], WHITE_COLOR)

    # If arm is missing, try to approximate forearm
    if arm:
        px = list(np.array(PX[9]) + 2.5 * (np.array(PX[0])-np.array(PX[9])))
        px[0], px[1] = int(px[0]), int(px[1])
        circle_segmentation(maskImage, PX[0], px, ratio = int(9./26.))


def draw_arm(image, maskImage, landmark_list, connections):
    image_rows, image_cols, _ = image.shape

    if not landmark_list or not connections:
        return

    PX = []
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        PX.append(landmark_px)

    # Draws the connections if the start and end landmarks are both visible.
    for i, connection in enumerate(connections):
        if len(connection) == 2:
            start_idx = connection[0]
            end_idx = connection[1]
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio = arm_ratio)

def countdown_start(time_factor):
    print(" ")
    print("The recording will begin in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor)
    print(" ")

def save_img(path_img, path_mask, name, i, image, mask):
    dim = (480, 640)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

    img_name = path_img + '/' + name + '_{}'.format(i)
    mask_name = path_mask + '/' + name + '_{}'.format(i)

    tt = str(time.asctime())
    img_name_save = (img_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.png')).replace(' ', '_')
    mask_name_save = (mask_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.png')).replace(' ', '_')

    cv2.imwrite(img_name_save, image)
    cv2.imwrite(mask_name_save, mask)

if __name__ == '__main__':
    # For webcam input:
    cap = cv2.VideoCapture(0)
    W, H = 480, 640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    data_points_to_record = 3000

    class_img = 1
    if class_img == 1:
        save_path = r'/home/eranbamani/Documents/data_PointProject/data_seg/image/Point'
        save_pathmask = r'/home/eranbamani/Documents/data_PointProject/data_seg/mask/Point'

        name = 'Point'
    else:
        save_path = r'/home/eranbamani/Documents/data_PointProject/data_seg/image/NoPoint'
        save_pathmask = r'/home/eranbamani/Documents/data_PointProject/data_seg/mask/Point'

        name = 'NoPoint'

    countdown_start(1)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      for i in tqdm(range(data_points_to_record)):
        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        maskImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        maskImage[:,:] = 0
        results_hand = hands.process(image)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results_pose = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results_pose.pose_landmarks:
                pose_landmarks = results_pose.pose_landmarks
                draw_arm(image, maskImage, pose_landmarks, arm_connections)
            else:
                print('No body visible!')

            if results_hand.multi_hand_landmarks:
                for a, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
                    if results_hand.multi_handedness[a].classification[0].label == 'Left':
                        draw_hand(image, maskImage, hand_landmarks, mp_hands.HAND_CONNECTIONS, arm = not results_pose.pose_landmarks)

        except:
            pass

        cv2.imshow('Original image', cv2.flip(image, 1))
        cv2.imshow('Mask image', cv2.flip(maskImage, 1))
        save_img(save_path, save_pathmask, name, i, cv2.flip(image, 1), cv2.flip(maskImage, 1))

        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    cv2.destroyAllWindows()
    exit(0)