import cv2
import numpy as np
import onnxruntime
import time
from utils_crop import face_align, trans_points2d, distance2bbox, distance2kps, nms_boxes,\
                        crop_image, softmax, calculate_distance_ratio, get_rotation_matrix,\
                        transform_keypoint, concat_frame, prepare_paste_back, paste_back


def get_face_analysis(det_face, landmark):

    def get_landmark(img, face):
        input_size = 192

        bbox = face["bbox"]
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = input_size / (max(w, h) * 1.5)
        aimg, M = face_align(img, center, input_size, _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        aimg = aimg.transpose(2, 0, 1)  # HWC -> CHW
        aimg = np.expand_dims(aimg, axis=0)
        aimg = aimg.astype(np.float32)

        # feedforward
        output = landmark.run(None, {"data": aimg})
        pred = output[0][0]

        pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= input_size[0] // 2

        IM = cv2.invertAffineTransform(M)
        pred = trans_points2d(pred, IM)

        return pred

    def face_analysis(img):
        input_size = 512

        im_ratio = float(img.shape[0]) / img.shape[1]
        if im_ratio > 1:
            new_height = input_size
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        det_img = (det_img - 127.5) / 128
        det_img = det_img.transpose(2, 0, 1)  # HWC -> CHW
        det_img = np.expand_dims(det_img, axis=0)
        det_img = det_img.astype(np.float32)

        # feedforward
        output = det_face.run(None, {"input.1": det_img})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        det_thresh = 0.5
        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        center_cache = {}
        for idx, stride in enumerate(feat_stride_fpn):
            scores = output[idx]
            bbox_preds = output[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = output[idx + fmc * 2] * stride
            height = input_size // stride
            width = input_size // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                num_anchors = 2
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        nms_thresh = 0.4
        keep = nms_boxes(pre_det, [1 for s in pre_det], nms_thresh)
        bboxes = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if bboxes.shape[0] == 0:
            return []

        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = dict(bbox=bbox, kps=kps, det_score=det_score)
            lmk = get_landmark(img, face)
            face["landmark_2d_106"] = lmk

            ret.append(face)

        src_face = sorted(
            ret,
            key=lambda face: (face["bbox"][2] - face["bbox"][0])
            * (face["bbox"][3] - face["bbox"][1]),
            reverse=True,
        )

        return src_face

    return face_analysis


def preprocess(img):
    img = img / 255.0
    img = np.clip(img, 0, 1)  # clip to 0~1
    img = img.transpose(2, 0, 1)  # HxWx3x1 -> 1x3xHxW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def src_preprocess(img):
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    max_dim = 1280
    if max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = 2
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def crop_src_image(models, img):
    face_analysis = models["face_analysis"]
    src_face = face_analysis(img)

    if len(src_face) == 0:
        print("No face detected in the source image.")
        return None
    elif len(src_face) > 1:
        print(f"More than one face detected in the image, only pick one face.")

    src_face = src_face[0]
    lmk = src_face["landmark_2d_106"]  # this is the 106 landmarks from insightface

    # crop the face
    crop_info = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125)

    lmk = landmark_runner(models, img, lmk)

    crop_info["lmk_crop"] = lmk
    crop_info["img_crop_256x256"] = cv2.resize(
        crop_info["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
    )
    crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512

    return crop_info


def landmark_runner(models, img, lmk):
    crop_dct = crop_image(img, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
    img_crop = crop_dct["img_crop"]

    img_crop = img_crop / 255
    img_crop = img_crop.transpose(2, 0, 1)  # HWC -> CHW
    img_crop = np.expand_dims(img_crop, axis=0)
    img_crop = img_crop.astype(np.float32)

    # feedforward
    net = models["landmark_runner"]
    output = net.run(None, {"input": img_crop})
    out_pts = output[2]

    # 2d landmarks 203 points
    lmk = out_pts[0].reshape(-1, 2) * 224  # scale to 0-224
    # _transform_pts
    M = crop_dct["M_c2o"]
    lmk = lmk @ M[:2, :2].T + M[:2, 2]

    return lmk


def extract_feature_3d(models, x):
    net = models["appearance_feature_extractor"]

    # feedforward
    output = net.run(None, {"img": x})
    f_s = output[0]
    f_s = f_s.astype(np.float32)

    return f_s


def get_kp_info(models, x):
    net = models["motion_extractor"]

    # feedforward
    output = net.run(None, {"img": x})
    pitch, yaw, roll, t, exp, scale, kp = output

    kp_info = dict(pitch=pitch, yaw=yaw, roll=roll, t=t, exp=exp, scale=scale, kp=kp)

    pred = softmax(kp_info["pitch"], axis=1)
    degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
    kp_info["pitch"] = degree[:, None]  # Bx1
    pred = softmax(kp_info["yaw"], axis=1)
    degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
    kp_info["yaw"] = degree[:, None]  # Bx1
    pred = softmax(kp_info["roll"], axis=1)
    degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
    kp_info["roll"] = degree[:, None]  # Bx1

    kp_info = {k: v.astype(np.float32) for k, v in kp_info.items()}

    bs = kp_info["kp"].shape[0]
    kp_info["kp"] = kp_info["kp"].reshape(bs, -1, 3)  # BxNx3
    kp_info["exp"] = kp_info["exp"].reshape(bs, -1, 3)  # BxNx3

    return kp_info


def stitching(models, kp_source, kp_driving):
    """conduct the stitching
    kp_source: Bxnum_kpx3
    kp_driving: Bxnum_kpx3
    """

    bs, num_kp = kp_source.shape[:2]

    kp_driving_new = kp_driving

    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    feat = np.concatenate(
        [kp_source.reshape(bs_src, -1), kp_driving.reshape(bs_dri, -1)], axis=1
    )

    # feedforward
    net = models["stitching"]
    output = net.run(None, {"input": feat})
    delta = output[0]

    delta_exp = delta[..., : 3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
    delta_tx_ty = delta[..., 3 * num_kp : 3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

    kp_driving_new += delta_exp
    kp_driving_new[..., :2] += delta_tx_ty

    return kp_driving_new


def warping_spade(models, feature_3d, kp_source, kp_driving):
    """get the image after the warping of the implicit keypoints
    feature_3d: Bx32x16x64x64, feature volume
    kp_source: BxNx3
    kp_driving: BxNx3
    """

    # feedforward
    net = models["warping_spade"]
    output = net.run(
            None,
            {
                "feature_3d": feature_3d,
                "kp_driving": kp_driving,
                "kp_source": kp_source,
            },
        )
    return output[0]


def predict(frame_id, models, x_s_info, R_s, f_s, x_s, img, pred_info):
    # calc_lmks_from_cropped_video
    frame_0 = pred_info['lmk'] is None
    if frame_0:
        face_analysis = models["face_analysis"]
        src_face = face_analysis(img)
        if len(src_face) == 0:
            print(f"No face detected in the frame")
            raise Exception(f"No face detected in the frame")
        elif len(src_face) > 1:
            print(f"More than one face detected in the driving frame, only pick one face.")
        src_face = src_face[0]
        lmk = src_face["landmark_2d_106"]
        lmk = landmark_runner(models, img, lmk)
    else:
        lmk = landmark_runner(models, img, pred_info['lmk'])
    pred_info['lmk'] = lmk

    # calc_driving_ratio
    lmk = lmk[None]
    c_d_eyes = np.concatenate(
        [
            calculate_distance_ratio(lmk, 6, 18, 0, 12),
            calculate_distance_ratio(lmk, 30, 42, 24, 36),
        ],
        axis=1,
    )
    c_d_lip = calculate_distance_ratio(lmk, 90, 102, 48, 66)
    c_d_eyes = c_d_eyes.astype(np.float32)
    c_d_lip = c_d_lip.astype(np.float32)

    # prepare_driving_videos
    img = cv2.resize(img, (256, 256))
    I_d = preprocess(img)

    # collect s_d, R_d, δ_d and t_d for inference
    x_d_info = get_kp_info(models, I_d)
    R_d = get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"])
    x_d_info = {
        "scale": x_d_info["scale"].astype(np.float32),
        "R_d": R_d.astype(np.float32),
        "exp": x_d_info["exp"].astype(np.float32),
        "t": x_d_info["t"].astype(np.float32),
    }

    if frame_0:
        pred_info['x_d_0_info'] = x_d_info

    x_d_0_info = pred_info['x_d_0_info']
    R_d_0 = x_d_0_info["R_d"]

    R_new = (R_d @ R_d_0.transpose(0, 2, 1)) @ R_s
    delta_new = x_s_info["exp"] + (x_d_info["exp"] - x_d_0_info["exp"])
    scale_new = x_s_info["scale"] * (x_d_info["scale"] / x_d_0_info["scale"])
    t_new = x_s_info["t"] + (x_d_info["t"] - x_d_0_info["t"])

    t_new[..., 2] = 0  # zero tz
    x_c_s = x_s_info["kp"]
    x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
        
    # with stitching and without retargeting
    x_d_new = stitching(models, x_s, x_d_new)

    out = warping_spade(models, f_s, x_s, x_d_new)
    # out = out["out"]
    out = out.transpose(0, 2, 3, 1)  # 1x3xHxW -> 1xHxWx3
    out = np.clip(out, 0, 1)  # clip to 0~1
    out = (out * 255).astype(np.uint8)  # 0~1 -> 0~255
    I_p = out[0]

    return I_p, pred_info

class LivePortraitWrapper():
    def __init__(self):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        appearance_feature_extractor = onnxruntime.InferenceSession("weights/appearance_feature_extractor.onnx", so)
        motion_extractor = onnxruntime.InferenceSession("weights/motion_extractor.onnx", so)
        warping_spade = onnxruntime.InferenceSession("weights/warping_spade.onnx", so)
        stitching_module = onnxruntime.InferenceSession("weights/stitching.onnx", so)
        landmark_run = onnxruntime.InferenceSession("weights/landmark.onnx", so)
        det_face = onnxruntime.InferenceSession("weights/retinaface_det_static.onnx", so)
        landmark = onnxruntime.InferenceSession("weights/face_2dpose_106_static.onnx", so)

        face_analysis = get_face_analysis(det_face, landmark)

        self.models = {
            "appearance_feature_extractor": appearance_feature_extractor,
            "motion_extractor": motion_extractor,
            "warping_spade": warping_spade,
            "stitching": stitching_module,
            "landmark_runner": landmark_run,
            "face_analysis": face_analysis,
        }
        self.maskpath = 'mask_template.png'
        mask_crop = cv2.imread(self.maskpath)
        self.mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGRA2BGR)
        self.flg_composite = False
        self.pred_info = {'lmk':None, 'x_d_0_info':None}

    def execute(self, imgpath, videopath):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = img[:, :, ::-1]  # BGR -> RGB
        src_img = src_preprocess(img)
        crop_info = crop_src_image(self.models, src_img)

        # prepare_source
        img_crop_256x256 = crop_info["img_crop_256x256"]
        I_s = preprocess(img_crop_256x256)

        x_s_info = get_kp_info(self.models, I_s)
        R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
        f_s = extract_feature_3d(self.models, I_s)
        x_s = transform_keypoint(x_s_info)

        capture = cv2.VideoCapture(videopath)
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        f_h, f_w = (512, 512 * 3) if self.flg_composite else src_img.shape[:2]

        # prepare for pasteback
        mask_ori = prepare_paste_back(self.mask_crop, crop_info["M_c2o"], dsize=(src_img.shape[1], src_img.shape[0]))

        video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (f_w, f_h))
        frame_id = 0
        while True:
            ret, frame = capture.read()
            if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
                break

            # inference
            img_rgb = frame[:, :, ::-1]  # BGR -> RGB
            a = time.time()
            I_p, self.pred_info = predict(frame_id, self.models, x_s_info, R_s, f_s, x_s, img_rgb, self.pred_info)
            b = time.time()
            print(f"frame_id={frame_id}, predict waste time {b-a} s")
            frame_id += 1

            if self.flg_composite:
                driving_img = concat_frame(img_rgb, img_crop_256x256, I_p)
            else:
                driving_img = paste_back(I_p, crop_info["M_c2o"], src_img, mask_ori)
            driving_img = driving_img[:, :, ::-1]  # RGB -> BGR

            # cv2.imshow('frame', driving_img)
            # key = cv2.waitKey(1)
            # if key == 27:  # ESC
            #     break

            video_writer.write(driving_img)
        
        capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    live_portrait_pipeline = LivePortraitWrapper()

    imgpath = '0.jpg'
    videopath = 'd0.mp4'
    live_portrait_pipeline.execute(imgpath, videopath)
