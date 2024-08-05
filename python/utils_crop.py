import numpy as np
import cv2
from math import sin, cos, acos, degrees


def _transform_img(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=None):
    """conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(
            img,
            M[:2, :],
            dsize=_dsize,
            flags=flags,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    """conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]


def parse_pt2_from_pt106(pt106, use_lip=True):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt_x(pts, use_lip=True):
    pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2


def parse_rect_from_landmark(
    pts,
    scale=1.5,
    need_square=True,
    vx_ratio=0,
    vy_ratio=0,
    use_deg_flag=False,
    **kwargs,
):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get("use_lip", True))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=np.float32)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = (
        center0 + ux * center1[0] + uy * center1[1]
    )  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = (
        center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    )  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle


def _estimate_similar_transform_from_pts(
    pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1, flag_do_rot=True, **kwargs
):
    """calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        use_lip=kwargs.get("use_lip", True),
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [
                [s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                [
                    -s * sintheta,
                    s * costheta,
                    tcy - s * (-sintheta * cx + costheta * cy),
                ],
            ],
            dtype=np.float32,
        )
    else:
        M_INV = np.array(
            [
                [s, 0, tgt_center[0] - s * center[0]],
                [0, s, tgt_center[1] - s * center[1]],
            ],
            dtype=np.float32,
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]


def crop_image(img, pts: np.ndarray, dsize=224, scale=1.5, vy_ratio=-0.1):
    M_INV, _ = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=True,
    )

    img_crop = _transform_img(img, M_INV, dsize)  # origin to crop
    pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=np.float32)])
    M_c2o = np.linalg.inv(M_o2c)

    ret_dct = {
        "M_o2c": M_o2c,  # from the original image to the cropped image 3x3
        "M_c2o": M_c2o,  # from the cropped image to the original image 3x3
        "img_crop": img_crop,  # the cropped image
        "pt_crop": pt_crop,  # the landmarks of the cropped image
    }

    return ret_dct

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def face_align(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    
    trans_M = np.array([[scale_ratio*cos(rot), -scale_ratio*sin(rot), output_size*0.5-center[0]*scale_ratio], 
                        [scale_ratio*sin(rot), scale_ratio*cos(rot), output_size*0.5-center[1]*scale_ratio], 
                        [0, 0, 1]], dtype=np.float32)
    M = trans_M[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)

    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def calculate_distance_ratio(
    lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6
) -> np.ndarray:
    return np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) / (
        np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps
    )


def get_rotation_matrix(pitch_, yaw_, roll_):
    """the input is in degree"""
    # transform to radian
    pitch = pitch_ / 180 * np.pi
    yaw = yaw_ / 180 * np.pi
    roll = roll_ / 180 * np.pi

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = np.ones([bs, 1], dtype=np.float32)
    zeros = np.zeros([bs, 1], dtype=np.float32)
    x, y, z = pitch, yaw, roll

    rot_x = np.concatenate(
        [ones, zeros, zeros, zeros, np.cos(x), -np.sin(x), zeros, np.sin(x), np.cos(x)],
        axis=1,
    ).reshape([bs, 3, 3])

    rot_y = np.concatenate(
        [np.cos(y), zeros, np.sin(y), zeros, ones, zeros, -np.sin(y), zeros, np.cos(y)],
        axis=1,
    ).reshape([bs, 3, 3])

    rot_z = np.concatenate(
        [np.cos(z), -np.sin(z), zeros, np.sin(z), np.cos(z), zeros, zeros, zeros, ones],
        axis=1,
    ).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.transpose(0, 2, 1)  # transpose


def transform_keypoint(kp_info: dict):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = kp_info["kp"]  # (bs, k, 3)
    pitch, yaw, roll = kp_info["pitch"], kp_info["yaw"], kp_info["roll"]

    t, exp = kp_info["t"], kp_info["exp"]
    scale = kp_info["scale"]

    bs = kp.shape[0]
    num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = kp.reshape(bs, num_kp, 3) @ rot_mat + exp.reshape(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed


def prepare_paste_back(mask_crop, crop_M_c2o, dsize):
    """prepare mask for later image paste back"""
    mask_ori = cv2.warpAffine(
        mask_crop, crop_M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR
    )
    mask_ori = mask_ori.astype(np.float32) / 255.0
    return mask_ori


def paste_back(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image"""
    dsize = (img_ori.shape[1], img_ori.shape[0])
    result = cv2.warpAffine(img_crop, M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
    result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(
        np.uint8
    )
    return result

def concat_frame(driving_img, src_img, I_p):
    h, w, _ = I_p.shape

    src_img = cv2.resize(src_img, (w, h))
    driving_img = cv2.resize(driving_img, (w, h))
    out = np.hstack((driving_img, src_img, I_p))

    return out

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def nms_boxes(boxes, scores, iou_thres):
    # Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    keep = []
    for i, box_a in enumerate(boxes):
        is_keep = True
        for j in range(i):
            if not keep[j]:
                continue
            box_b = boxes[j]
            iou = bb_intersection_over_union(box_a, box_b)
            if iou >= iou_thres:
                if scores[i] > scores[j]:
                    keep[j] = False
                else:
                    is_keep = False
                    break

        keep.append(is_keep)

    return np.array(keep).nonzero()[0]

def softmax(x, axis=None):
    max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=axis, keepdims=True)
    f_x = e_x / sum
    return f_x
