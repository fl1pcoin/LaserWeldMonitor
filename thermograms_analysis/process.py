import numpy as np
import cv2
import scipy.signal as sig
from typing import List, Dict, Tuple, Union
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from tqdm import tqdm


R = 40  # radius to filter arround the welding zone
DX_REFL, DY_REFL, R_REFL = -80, -10, 20  # parameters of reflection to remove
DX_TR, DY_TR, R_TR = 40, -40, 15  # parameters of trace to remove


def min_loc_LoG(img, k_size = 9, sigma = 1.8):
    """
    Perform min-loc-LoG filtering of grayscale image img
    Sungho K. Min-local-LoG Filter for Detecting Small Targets in 
    Cluttered Background // Electronics Letters. 
    – 2011. – Vol. 47. – № 2. – P. 105-106. DOI: 10.1049/el.2010.2066.

    sigma - std of gaussian
    k_size - size of kernel
    """
    x = np.arange(k_size).reshape(1, k_size)
    y = np.arange(k_size).reshape(k_size, 1)
    # generate fE (positive X)
    fE = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fE[fE > 0] = fE[fE > 0] / fE[fE > 0].sum()
    fE[fE < 0] = fE[fE < 0] / (-fE[fE < 0].sum())
    # generate fS (positive Y)
    fS = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fS[fS > 0] = fS[fS > 0] / fS[fS > 0].sum()
    fS[fS < 0] = fS[fS < 0] / (-fS[fS < 0].sum())
    # generate fW
    x = - np.fliplr(x)
    fW = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fW[fW > 0] = fW[fW > 0] / fW[fW > 0].sum()
    fW[fW < 0] = fW[fW < 0] / (-fW[fW < 0].sum())
    # generate fN
    y = - np.flipud(y)
    fN = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fN[fN > 0] = fN[fN > 0] / fN[fN > 0].sum()
    fN[fN < 0] = fN[fN < 0] / (-fN[fN < 0].sum())
    # perform 2D convolution with kernels
    def move(img, x, y):
        move_matrix = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv2.warpAffine(img, move_matrix, dimensions)

    Ie = sig.convolve2d(move(img, 4, 0), fE, mode = "same")
    Is = sig.convolve2d(move(img, 0, 4), fS, mode = "same")
    Iw = sig.convolve2d(move(img, -4, 0), fW, mode = "same")
    In = sig.convolve2d(move(img, 0, -4), fN, mode = "same")
    f = np.dstack((Ie, Is, Iw, In))
    fmap = np.min(f, axis = 2)
    #return (fmap / fmap.max() * 255).astype(np.uint8)
    return fmap

        
def detect_spatters(frame):
    filtered = min_loc_LoG(frame, 9)
    filtered = ((filtered > 2) * 255).astype(np.uint8)
    c, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3]]) for i,_ in enumerate(c)]
    contours = np.array(contours)
    if not len(contours):
        return ()
    wh = contours[:, 2:4] - contours[:, :2]
    contours[:, :2] = contours[:, :2] + wh / 2
    contours[:, 2:4] = wh
    contours = contours[np.all(wh < 5, axis=1)]
    return contours


def detect_welding_zone(frame):
    filtered = ((frame > 20) * 255).astype(np.uint8)
    c, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3]]) for i,_ in enumerate(c)]
    center = np.array(center)
    wh = center[:, 2:4] - center[:, :2]
    center[:, :2] = center[:, :2] + wh / 2
    center[:, 2:4] = wh
    areas = wh[:, 0] * wh[:, 1]
    max_id = areas.argmax()
    center = center[max_id]
    if areas[max_id] > 10:
        return center, cv2.fitEllipse(c[max_id])
    else:
        raise ValueError("Can't detect welding zone")


def filter_spatters(boxes, center, R):
    x, y = center[:2]
    contours = boxes[(boxes[:, 0] - x) ** 2 + (boxes[:, 1] - y) ** 2 > R ** 2]
    return contours


def remove_reflection(boxes, center, dx, dy, r):
    x, y = center[:2]
    x += dx
    y += dy
    contours = boxes[(boxes[:, 0] - x) ** 2 + (boxes[:, 1] - y) ** 2 > r ** 2]
    return contours


class NearestTracker:
    """
This class is used to track points between frames in a video or a sequence of images. It maintains a dictionary of tracked points and performs operations such as matching previously tracked points with new detections, deleting unmatched tracks, and assigning new IDs to unmatched detections.

Methods:
    __init__(self):
        Initializes the class and sets up the necessary attributes. It initializes a dictionary to keep track of objects, sets the maximum id to 0, and the maximum distance to 30.

    step(self, pts: numpy.ndarray) -> dict:
        Performs a step in tracking points between frames. It first checks if there are points being tracked already. If not, it assigns IDs to the newly detected points. Next, it matches the previously tracked points with the new detections. If some points are not matched, they are considered for deletion or assignment of new IDs. Finally, it updates the tracked points, removes unmatched tracks, and assigns new IDs to unmatched detections.

        Args:
            pts: An array of new points detected in the current frame.
        
        Returns:
            A deepcopy of the dictionary containing the tracked points after the matching, deletion and addition process. Each key in the dictionary is the ID of the point and the value is its corresponding point."""
    def __init__(self):
        """
    Initializer for the class.

    This method initializes the class and sets up the necessary attributes. It initializes a 
    dictionary to keep track of objects, sets the maximum id to 0, and the maximum distance to 30.

    Args:
        self (object): The instance of the class.

    Returns:
        None
    """
        self.tracked = {}
        self.max_id = 0
        self.max_d = 30
    
    def step(self, pts):
        """
        This method performs a step in tracking points between frames. It first checks if there are points being tracked already. If not, it assigns IDs to the newly detected points. Next, it matches the previously tracked points with the new detections. If some points are not matched, they are considered for deletion or assignment of new IDs. Finally, it updates the tracked points, removes unmatched tracks, and assigns new IDs to unmatched detections.

        Args:
            self (object): Self reference which points to the instance calling the method.
            pts (numpy.ndarray): An array of new points detected in the current frame.

        Returns:
            dict: A deepcopy of the dictionary containing the tracked points after the matching, deletion and addition process. Each key in the dictionary is the ID of the point and the value is its corresponding point.
        """
        if not len(self.tracked):
            self.tracked = {i: pt for i, pt in enumerate(pts)}
            self.max_id = len(self.tracked)
            return deepcopy(self.tracked)
        
        ids_not_matched = []  # list of ids not matched from previous step
        pts_not_matched = []  # list of new kpts to assign new ids

        # Match tracked pts with new detections
        new_pts = np.expand_dims(pts[:, :2], 1)
        pts_ = np.array(list(self.tracked.values()))
        old_pts = np.expand_dims(pts_[:, :2], 0)
        diff = np.sqrt(np.square(new_pts - old_pts).sum(axis=-1))
        a = (diff < self.max_d).astype(int)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            x, y = linear_sum_assignment(diff)
            matched_indices = np.array(list(zip(x, y)))
        
        # Select unmatched detections and tracks
        for d, det in enumerate(pts):  # iterate over detected kpts
            if d not in matched_indices[:, 0]:
                pts_not_matched.append(det)  # collect unmatched new points to assign new ids

        for t, (key, value) in enumerate(self.tracked.items()):  # iterate over tracked kpts
            if t not in matched_indices[:, 1]:
                ids_not_matched.append(key)  # collect unmatched old points to discard

        # Filter matched detections by distance
        matched = []
        keys = list(self.tracked.keys())
        for x, y in matched_indices:
            if diff[x, y] > self.max_d:
                pts_not_matched.append(pts[x])
                ids_not_matched.append(keys[y])
            else:
                matched.append((x, y))

        # Replace matched detections
        if len(matched) != 0:
            for x, y in matched:
                self.tracked[list(self.tracked.keys())[y]] = pts[x]

        # Remove unmatched tracks
        for id in ids_not_matched:
            self.tracked.pop(id)
        # Add unmatched detections
        for det in pts_not_matched:
            self.tracked[self.max_id] = det
            self.max_id += 1
        
        return deepcopy(self.tracked)


class Tracker:
    """
    This is a class for tracking keypoints in video frames. It provides methods for initializing instance variables, 
    computing the angle between two points with respect to a vertex, computing the Euclidean distance between two points, 
    matching points, and updating keypoints and tracks.

    Methods:
    - __init__(self): Initializes instance variables to their default values. These include `max_d`, `delta_angle`, 
    `tracked`, `kpts`, and `max_id`.
    
    - __compute_angle(pt1, pt2, ver): Computes the angle between two points with respect to a vertex. 
    It first calculates the cosine of the angle using the dot product and then converts this to degrees.

    - __compute_distance(pt1, pts2): Computes the Euclidean distance between two points in a multi-dimensional space. 
    The computation is vectorized over pts2 for efficiency.
    
    - __match_pts(self, pts, pt1, pt2, used_ids): Computes the optimal and minimum distances between given points 
    and a list of detected keypoints. It returns the ID and difference value of the optimal match and the ID and 
    minimum distance value from the keypoints list.
    
    - step(pts): Accumulates key points from the last three frames, updates existing tracks, matches new detections 
    with existing tracks, combines detections into new tracks, and updates tracks and detections from previous iterations. 
    It returns a tuple if there are no points or less than three points. If there are three or more points, it returns 
    a dictionary where the keys are the track IDs and the values are the corresponding updated tracks."""
    def __init__(self):
        """
        Initializes a new instance of the class.

        This method initializes several instance variables to their default values. These variables include `max_d` which is set to 30 and represents the maximum distance between joint points, 
        `delta_angle` which is set to 20 and represents the maximum delta angle 180 +- to match points (in degrees), `tracked` which is set to an empty dictionary and is used to store tracked points (last 2 states stored),
        `kpts` which is set to an empty list and used to store keypoints from the last 3 frames, and `max_id` which is set to 0 and used to store the maximum assigned ID.

        Args:
            self: A reference to the instance of the class.

        Returns:
            None
        """
        self.max_d = 30  # maximum distance between joint points
        self.delta_angle = 20 # maximum delta angle 180 +- to match pts (in degrees)
        self.tracked = {}  # dict with tracked points (last 2 states stored)
        self.kpts = []  # kpts from last 3 frames
        self.max_id = 0  # max assigned ID

    @staticmethod
    def __compute_angle(pt1: np.ndarray, pt2: np.ndarray, ver: np.ndarray) -> float:
        """
    Computes the angle between two points with respect to a vertex.

    This method calculates the angle between the two points `pt1` and `pt2` using the vertex `ver`. 
    It does this by first calculating the cosine of the angle using the dot product, then converting
    this to degrees.

    Args:
        pt1 (np.ndarray): The first point as a numpy array. This point is expected to have at least two coordinates.
        pt2 (np.ndarray): The second point as a numpy array. This point is expected to have at least two coordinates.
        ver (np.ndarray): The vertex point as a numpy array. This point is expected to have at least two coordinates.

    Returns:
        float: The angle in degrees between `pt1` and `pt2` with respect to `ver`.
    """
        cosine = np.dot(pt1[:2] - ver[:2], pt2[:2] - ver[:2]) / (np.linalg.norm(pt1[:2] - ver[:2]) * np.linalg.norm(pt2[:2] - ver[:2]))
        angle = np.degrees(np.arccos(cosine - np.sign(cosine) * 10e-6))
        return angle

    @staticmethod
    def __compute_distance(pt1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        return np.sqrt(np.square(pts2 - pt1).sum(axis=-1))
    
    def __match_pts(self, pts: Union[np.ndarray, List[np.ndarray]], pt1: np.ndarray, pt2: np.ndarray,
                    used_ids: List[int]) -> Tuple[Union[int, None], float, Union[int, None], float]:
        """
    Computes the optimal and minimum distances between given points and a list of detected keypoints.

    Args:
        self: Instance of the class.
        pts (Union[np.ndarray, List[np.ndarray]]): A numpy array or list of detected keypoints.
        pt1 (np.ndarray): The first point for distance computation.
        pt2 (np.ndarray): The second point for distance computation.
        used_ids (List[int]): A list of IDs that are already used and are to be skipped in the computation.

    Returns:
        Tuple[Union[int, None], float, Union[int, None], float]: A tuple containing the ID and difference value of 
        the optimal match, and the ID and minimum distance value from the keypoints list, respectively. If no optimal 
        match is found, the ID is returned as None.
    """
        d1 = self.__compute_distance(pt1, pt2)  # compute distance between them
        # Init optimal
        optimal_id = None
        best_diff = 181.0

        # Init minimum
        min_id = None
        min_d = 1000

        # Iterate over detected kpts
        for det_id, det in enumerate(pts):
            # Skip if this detection was used
            if det_id in used_ids:
                continue
            d2 = self.__compute_distance(det, pt2)  # compute distance between last pt in track and detection
            # Find minimum distance
            if d2 < min_d:
                min_d = d2
                min_id = det_id
            # Filter detection by specific conditions (distance)
            if d2 > self.max_d:  # distance is less than max
                continue
            if min(d1, d2) / max(d1, d2) < 0.5:
                continue  # distance is nearly same as distance between previous kpts (optional)
            
            # Filter detection by angle
            angle = self.__compute_angle(pt1, det, pt2)  # compute angle between three pts
            diff = 180 - angle
            if diff > self.delta_angle:
                continue  # three pts don't form a straight line

            # If all filters passed, identify the most optimal match by criterion
            # TODO: try to change criterion
            if diff < best_diff:
                best_diff = diff
                optimal_id = det_id
        return optimal_id, best_diff, min_id, min_d

    def step(self, pts: Union[np.ndarray, None]) -> Union[Tuple, Dict[int, List[List[np.ndarray]]]]:
        # Accumulate kpts from 3 last frames
        if not len(pts):
            try:
                self.kpts.pop(0)
            except:
                pass
            return ()
        self.kpts.append(pts)
        if len(self.kpts) < 3:
            return ()
        
        new_tracks = []  # list for matched tracks
        used_ids_prev = []  # used pts from previous frame
        used_ids = []  # used pts from current frame
        unmatched_ids = []  # unmatched tracks

        # Match new detections with existed tracks
        if len(self.tracked):
            for id, track in self.tracked.items():  # iterate over existing tracks
                pt1 = track[-2][:2]  # get 2 last points from track
                pt2 = track[-1][:2]
                # Match
                optimal_id, _, min_id, min_d = self.__match_pts(pts[:, :2], pt1, pt2, used_ids)
                # Update track if match found
                if optimal_id is not None:
                    self.tracked[id].pop(0)
                    self.tracked[id].append(pts[optimal_id])
                    used_ids.append(optimal_id)
                elif min_id is not None and min_d < 2:
                    self.tracked[id].pop(0)
                    self.tracked[id].append(pts[min_id])
                    used_ids.append(min_id)
                else:
                    unmatched_ids.append(id)
            
            for id in unmatched_ids:
                self.tracked.pop(id)

        # Combine detections into new tracks
        pts1 = self.kpts[0]  # kpts from prev prev frame
        pts2 = self.kpts[1]  # kpts from prev frame
        if len(pts1) and len(pts2):
            # Iterate over prev prev pts
            for id1, pt1 in enumerate(pts1):
                optimal_id_2 = None
                optimal_id_3 = None
                best_diff_2 = 181.0
                # Iterate over prev kpts
                for id2, pt2 in enumerate(pts2):
                    if id2 in used_ids_prev:
                        continue  # skip if already matched
                    if self.__compute_distance(pt1[:2], pt2[:2]) > self.max_d:
                        continue  # check distance condition
                    optimal_id, best_diff, _, _ = self.__match_pts(pts[:, :2], pt1[:2], pt2[:2], used_ids)  # find best match
                    if optimal_id is None:
                        continue  # next if match not found
                    # Find optimal combination
                    if best_diff < best_diff_2:
                        optimal_id_3 = optimal_id
                        optimal_id_2 = id2
                        best_diff_2 = best_diff
                
                if (optimal_id_3 is not None) and (optimal_id_2 is not None):
                    # Update new combined tracks and used ids
                    new_tracks.append([pt1, pts2[optimal_id_2], pts[optimal_id_3]])
                    used_ids_prev.append(optimal_id_2)
                    used_ids.append(optimal_id_3)

        # Update tracks
        for track in new_tracks:
            self.tracked[self.max_id] = track
            self.max_id += 1

        # Update detections from previous iterations
        prev_kpts = []
        cur_kpts = []
        for id, pt in enumerate(pts):
            if id not in (used_ids):
                cur_kpts.append(pt)
        for id, pt in enumerate(pts2):
            if id not in (used_ids_prev):
                prev_kpts.append(pt)
        self.kpts = [prev_kpts, cur_kpts]
        return self.tracked.copy()


class FeatureExtractor:
    """
This class is designed to store, update, and compute metrics related to object tracking in a welding zone.

Methods:
    __init__(w_size: int = 10):
        Initializes the class with a metrics dictionary and sets initial variables.
        
    compute_results(use_interpolation: bool = False) -> Dict[str, float]:
        Computes overall metrics for the object based on its attributes.
        
    append(tracks: Dict[int, List[List[np.ndarray]]], t_frame: np.ndarray, center: np.ndarray):
        Appends various calculated metrics to their respective lists in the 'self.metrics' dictionary."""
    def __init__(self, w_size: int = 10):
        """
    Initializes the class with a metrics dictionary and sets initial variables.

    This method initializes the class by creating a dictionary to store metrics, sets the last tracks and 
    the last temperature frame to None, and defines the number of frames to interpolate.

    Args:
        w_size (int, optional): The number of frames to interpolate. Defaults to 10.

    Returns:
        None
    """
        self.metrics = {'total_spatters': [], 'velocity': [], 'size': [], 'temp': [], 'cooling_speed': [], 
                        'appearance_rate': [], 'n_spatters': [], 'welding_zone_temp': []}  # dict to store metrics
        self.last_tracks = None
        self.last_t_frame = None
        self.frames_to_interpolate = w_size

    def compute_results(self, use_interpolation: bool=False) -> Dict[str, float]:
        """
    Compute overall metrics of the object.

    This method computes overall metrics for the object based on its attributes. It divides the data into four zones and
    calculates the mean, maximum, and minimum values for each zone. If the use_interpolation parameter is set to True,
    the method interpolates the data before computing the metrics.

    Args:
        use_interpolation (bool): A flag that, if set to True, enables data interpolation before metrics computation.
                                   Defaults to False.

    Returns:
        Dict[str, float]: A dictionary where keys are the names of the metrics and the values are the computed results.
    """
        total = len(self.metrics['total_spatters'])
        chunk_size = total // 4
        overall_metrics = {}
        if not use_interpolation:
            for key in self.metrics.keys():
                for i, zone in enumerate(('A', 'B', 'C', 'D')):
                    data = np.array(self.metrics[key][i * chunk_size : (i + 1) * chunk_size])
                    if key == 'total_spatters':
                        overall_metrics[f"{key}_{zone}"] = int(data.max() - data.min())
                    elif key in ('appearance_rate', 'n_spatters'):
                        overall_metrics[f"mean_{key}_{zone}"] = float(data.mean())
                        overall_metrics[f"max_{key}_{zone}"] = float(data.max())
                        overall_metrics[f"min_{key}_{zone}"] = float(data[data != 0].min()) if data.sum() != 0 else 0
                    else:
                        overall_metrics[f"mean_{key}_{zone}"] = float(data[data != 0].mean())
                        overall_metrics[f"max_{key}_{zone}"] = float(data[data != 0].max())
                        overall_metrics[f"min_{key}_{zone}"] = float(data[data != 0].min())
        else:
            interp_size = chunk_size // self.frames_to_interpolate  # chunk size inside section
            for key in self.metrics.keys():
                for i, zone in enumerate(('A', 'B', 'C', 'D')):
                    data = np.array(self.metrics[key][i * chunk_size : (i + 1) * chunk_size])  # metric inside zone
                    if key == 'total_spatters':
                        overall_metrics[f"{key}_{zone}"] = []
                    else:
                        overall_metrics[f"mean_{key}_{zone}"] = []
                        overall_metrics[f"max_{key}_{zone}"] = []
                        overall_metrics[f"min_{key}_{zone}"] = []
                    for j in range(interp_size):
                        chunk = data[j * self.frames_to_interpolate : (j + 1) * self.frames_to_interpolate]
                        if key == 'total_spatters':
                            overall_metrics[f"{key}_{zone}"].append(int(chunk.max() - chunk.min()))
                        elif key in ('appearance_rate', 'n_spatters'):
                            overall_metrics[f"mean_{key}_{zone}"].append(float(chunk.mean()))
                            overall_metrics[f"max_{key}_{zone}"].append(float(chunk.max()))
                            overall_metrics[f"min_{key}_{zone}"].append(float(chunk[chunk != 0].min()) if chunk.sum() != 0 else 0)
                        else:
                            overall_metrics[f"mean_{key}_{zone}"].append(float(chunk[chunk != 0].mean()) if chunk.sum() != 0 else 0)
                            overall_metrics[f"max_{key}_{zone}"].append(float(chunk[chunk != 0].max()) if chunk.sum() != 0 else 0)
                            overall_metrics[f"min_{key}_{zone}"].append(float(chunk[chunk != 0].min()) if chunk.sum() != 0 else 0)

        return overall_metrics

    def append(self, tracks: Dict[int, List[List[np.ndarray]]], t_frame: np.ndarray, center: np.ndarray) -> None:
        """
    Appends various calculated metrics to their respective lists in the 'self.metrics' dictionary. 

    This method calculates various metrics such as the temperature of the welding zone, total number of spatters,
    number of spatters per frame, mean size of spatters per frame, mean temperature of spatters per frame,
    mean cooling speed of spatters per frame, mean velocity of spatters per frame, and the number of new spatters per frame.
    These metrics are then appended to their respective lists in the 'self.metrics' dictionary. It also updates 'last_t_frame' and
    'last_tracks' for use in the next iteration.

    Args:
        tracks (Dict[int, List[List[np.ndarray]]]): A dictionary where each key-value pair represents a track.
        t_frame (np.ndarray): A 2D numpy array representing the temperature frame.
        center (np.ndarray): A 1D numpy array representing the coordinates of the center of the tracking area.

    Returns:
        None
    """
        if len(tracks):
            # Temperature of welding zone (max)
            x, y, w, h = center
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            self.metrics['welding_zone_temp'].append(t_frame[y1:y2, x1:x2].max())
            # Total number of spatters
            self.metrics['total_spatters'].append(max(list(tracks.keys())))  # compute accumulated number of spatters
            # Number of spatters per frame
            self.metrics['n_spatters'].append(len(tracks))
            # Mean size of spatters per frame
            pts = np.array(list(tracks.values()))  # collect bboxes of all kpts on the previous 3 frames (shape: )
            cur_pts = pts[:, 0, :]  # get kpts on the current frame
            areas = cur_pts[:, 2] * cur_pts[:, 3]  # save size of current spatters
            self.metrics['size'].append(areas.mean())
            # Mean temperature of spatters per frame
            # Mean cooling speed of spatters per frame
            temp = []
            cool_speed = []
            for key, value in tracks.items():
                bboxes = np.array(value.copy())
                bboxes[:2, :2] = bboxes[:2,:2] - bboxes[:2,2:4] / 2
                bboxes[:2,2:4] = bboxes[:2,2:4] + bboxes[:2,:2]
                bboxes = bboxes.astype(int)
                x1, y1, x2, y2 = bboxes[0]
                cur_temp = t_frame[y1:y2, x1:x2].mean()
                temp.append(cur_temp)
                x1, y1, x2, y2 = bboxes[1]
                prev_temp = self.last_t_frame[y1:y2, x1:x2].mean()
                cool_speed.append(np.abs(prev_temp - cur_temp))
            
            self.metrics['temp'].append(np.array(temp).mean())
            self.metrics['cooling_speed'].append(np.array(cool_speed).mean())
            # Mean velocity of spatters per frame
            vel = np.sqrt(np.square(pts[:, 0, :2] - pts[:, 1, :2]).sum(axis=-1)).mean()
            self.metrics['velocity'].append(vel)
            # Number of new spatters per frame (appearance rate)
            if self.last_tracks is not None:
                new_ids = set(tracks.keys()) - set(self.last_tracks.keys())
                self.metrics['appearance_rate'].append(max(0, len(new_ids)))
            else:
                self.metrics['appearance_rate'].append(0)
            self.last_tracks = tracks.copy()
        else:
            for key in self.metrics.keys():
                self.metrics[key].append(0)
            self.last_tracks = None

        self.last_t_frame = t_frame  # save for the next iteration
        #print(self.metrics)
             

def process_thermogram(path: str, w_size: int) -> int:
    frames = np.load(path)
    temp_frames = frames.copy()
    # t_min = 1000
    # t_max = 2000
    # frames = np.clip(frames, t_min, t_max)
    # frames -= t_min
    # frames = frames / (t_max - t_min)
    t_min = frames.min()
    t_max = frames.max()
    frames -= t_min
    frames = frames / (t_max - t_min)
    frames *= 255
    frames = frames.astype(np.uint8)

    tracker = Tracker()

    scale = 2
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #recorder = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 512))

    fe = FeatureExtractor(w_size)

    k = 0

    for frame, t_frame in zip(frames, temp_frames):
        #print(k)
        k += 1
        pts = np.zeros((frame.shape[0] * scale, frame.shape[1] * scale) + (3,))  # for drawing

        try:
            center, ellips = detect_welding_zone(frame)
            frame = cv2.ellipse(frame, ellips, (255,), 1)
        except:
            continue

        boxes = detect_spatters(frame)
        try:
            boxes = filter_spatters(boxes, center, R)
            #boxes = remove_reflection(boxes, center, DX_REFL, DY_REFL, R_REFL)
            #boxes = remove_reflection(boxes, center, DX_TR, DY_TR, R_TR)
            x, y, w, h = center * scale
            pts = cv2.circle(pts, (int(x), int(y)), R * scale, (130, 130, 130), 1)
            #pts = cv2.circle(pts, (int(x + DX_REFL * scale), int(y + DY_REFL * scale)), R_REFL * scale, (255, 0, 0), 1)
            #frame = cv2.circle(frame, (int(x / scale + DX_REFL), int(y / scale + DY_REFL)), R_REFL, (255, ), 1)
            #pts = cv2.circle(pts, (int(x + DX_TR * scale), int(y + DY_TR * scale)), R_TR * scale, (0, 255, 0), 1)
            #frame = cv2.circle(frame, (int(x / scale + DX_TR), int(y / scale + DY_TR)), R_TR, (255, ), 1)
        except:
            pass

        tracked = tracker.step(boxes)

        fe.append(tracked, t_frame, center)

        # for pt in boxes:
        #     x, y, w, h = pt * scale
        #     pts = cv2.circle(pts, (int(x), int(y)), int((w + h) / 2), (0, 0, 255), 1)    
        

        if len(tracked) > 0:
            for track in tracked.values():
                pt1, pt2, pt3 = track
                x1, y1 = pt1[:2] * scale
                x2, y2 = pt2[:2] * scale
                x3, y3 = pt3[:2] * scale
                pts = cv2.line(pts, (x1, y1), (x2, y2), (255, 255, 255), 1)
                pts = cv2.line(pts, (x3, y3), (x2, y2), (255, 255, 255), 1)
                pts = cv2.circle(pts, (int(x3), int(y3)), 4, (0, 0, 255), 1)


        pts = cv2.putText(pts, f"Number of spatters {tracker.max_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 1)
        
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        frame = np.concatenate((frame, pts), axis=1).astype(np.uint8)
        # cv2.imshow('thermogram', frame)
        # if cv2.waitKey(0) == ord('q'):
        #    break
        #recorder.write(frame)

    #recorder.release()
    #cv2.destroyAllWindows()
    #print(fe.compute_results())
    return fe.compute_results(True).copy()

import os

W_SIZE = 10

paths = os.listdir('data/')
counts = {}

for path in tqdm(paths):
   counts[path] = process_thermogram(f"data/{path}", W_SIZE)

#process_thermogram('data/thermogram_11.npy', 10)

import json

with open(f"metrics_{W_SIZE}.json", 'w') as f:
    json.dump(counts, f)

# print(counts)