from __future__ import division
import numpy as np
import cv2
import onnxruntime


def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def inverse3x3(m):
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )

    invdet = 1 / det

    out = m.copy()

    out[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet
    out[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet
    out[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet
    out[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet
    out[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet

    return out


def transform_pixel_fp(pt, center, scale, output_size, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = inverse3x3(t)
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)

    return new_pt[:2].astype(np.float32)


def center_crop(img, center, scale, output_size):
    center_new = center.copy()
    # Upper left point
    ul = np.array(
        transform_pixel_fp([0, 0], center_new, scale, output_size, invert=1)
    ).astype(int)
    # Bottom right point
    br = np.array(
        transform_pixel_fp(output_size, center_new, scale, output_size, invert=1)
    ).astype(int)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]  #
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]

    # For scipy version > 1.2.1
    new_img = new_img.astype(np.uint8)
    new_img = cv2.resize(new_img, dsize=output_size, interpolation=cv2.INTER_LINEAR)
    return new_img


class PostProcessor:
    def __init__(self):
        super(PostProcessor, self).__init__()

    def get_preds(self, scores, offset, mask=False):
        """
        get predictions from score maps in torch Tensor
        return type: torch.LongTensor
        """
        # NOTE: assume scores.ndim == 4
        scores_spatial = scores.reshape(scores.shape[0], scores.shape[1], -1)
        maxval = np.max(scores_spatial, 2)
        idx = np.argmax(scores_spatial, 2)

        maxval = maxval.reshape(
            scores.shape[0], scores.shape[1], 1
        )  # [1, n_landmarks, 1]

        idx = idx.reshape(scores.shape[0], scores.shape[1], 1)  # [1, n_landmarks, 1]
        preds = np.tile(idx, (1, 1, 2)).astype(
            np.float32
        )  # [1, n_landmarks, 2] where same location idx in 2 dimension. values range [0, 63]

        x_offset, y_offset = offset

        x_offset = x_offset.reshape(
            scores.shape[0], scores.shape[1], -1
        )  # [1, n_landmarks, 64]
        x_mask = np.eye(x_offset.shape[-1])[idx[:, :, 0]].astype(bool)
        x_offset = x_offset[x_mask].reshape(
            scores.shape[0], scores.shape[1]
        )  # 각 채널별 (landmark별) max score에 해당하는 pixel의 x_offset값을 뽑음

        y_offset = y_offset.reshape(scores.shape[0], scores.shape[1], -1)
        y_mask = np.eye(y_offset.shape[-1])[idx[:, :, 0]].astype(bool)
        y_offset = y_offset[y_mask].reshape(
            scores.shape[0], scores.shape[1]
        )  # 각 채널별 (landmark별) max score에 해당하는 pixel의 y_offset값을 뽑음

        preds[:, :, 0] = (preds[:, :, 0]) % scores.shape[
            3
        ] + x_offset  # x 좌표. value ranges from [-a, 8). ( a can be 0 or smaller than 0 )
        preds[:, :, 1] = (
            np.floor((preds[:, :, 1]) / scores.shape[2]) + y_offset
        )  # y 좌표. value ranges from [-a, 8). ( a can be 0 or smaller than 0)

        if mask:
            pred_mask = np.tile(maxval > 0, (1, 1, 2)).astype(np.float32)
            preds *= pred_mask

        return preds

    def transform_preds(self, coords, center, scale, output_size):
        for p in range(coords.shape[0]):
            # NOTE: assume float dtype
            coords[p, 0:2] = (
                transform_pixel_fp(coords[p, 0:2], center, scale, output_size, 1, 0) - 1
            )

        return coords

    def decode_preds(self, output, center, scale, res, mask=False, offset=None):
        coords = self.get_preds(output, offset, mask)
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = self.transform_preds(coords[i], center[i], scale[i], res)

        return preds


class FaceLandmarker:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file
        self.img_size = [256, 256]
        self.face_crop_pad_ratio_w = 0.2  # for face crop width padding
        self.face_crop_pad_ratio_h = 0.05  # for face crop height padding
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mask = False

    def prepare(self, use_gpu, ctx=0):
        if use_gpu:
            self.ort_session = onnxruntime.InferenceSession(self.onnx_file)
            self.ort_session.set_providers(
                ["CUDAExecutionProvider"], [{"device_id": ctx}]
            )
        else:
            sessionOptions = onnxruntime.SessionOptions()
            sessionOptions.intra_op_num_threads = 1
            sessionOptions.inter_op_num_threads = 1
            self.ort_session = onnxruntime.InferenceSession(
                self.onnx_file, sess_options=sessionOptions
            )
            print(
                "landmark intra_op_num_threads {} inter_op_num_threads {}".format(
                    sessionOptions.intra_op_num_threads,
                    sessionOptions.inter_op_num_threads,
                )
            )

        self.input_name = self.ort_session.get_inputs()[0].name  # 'data'

        self.outputs = [
            self.ort_session.get_outputs()[i].name
            for i in range(len(self.ort_session.get_outputs()))
        ]

        img = np.zeros((3, self.img_size[0], self.img_size[1]))
        input_blob = np.expand_dims(img, axis=0).astype(np.float32)  # NCHW

        out = self.ort_session.run(
            self.outputs, input_feed={self.input_name: input_blob}
        )

        # post processing setups
        self.pp = PostProcessor()

    def crop_face_from_frame(self, frame, face_box):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        x = face_box[0]
        y = face_box[1]
        w = face_box[2] - face_box[0]
        h = face_box[3] - face_box[1]
        pad_w = w * self.face_crop_pad_ratio_w
        pad_h = h * self.face_crop_pad_ratio_h
        # pad & clip to frame's size
        xmax = int(min(x + w + pad_w, frame_width - 1))
        ymax = int(min(y + h + pad_h, frame_height - 1))
        xmin = int(max(x - pad_w, 0))
        ymin = int(max(y - pad_h, 0))
        cropped = frame[ymin:ymax, xmin:xmax, :].copy()  # clone
        return cropped, (xmin, ymin, xmax, ymax)

    def run(self, frame, face_boxes):
        """
        frame: bgr order
        face_boxes: [# boxes, 4]
        """
        face_landmarks = []
        lmk_scores = []
        scores = None

        for face_box in face_boxes:
            # NOTE: crop face
            face_bgr, face_crop_coord = self.crop_face_from_frame(frame, face_box)

            # NOTE: BGR2RGB
            face_rgb = face_bgr[:, :, ::-1]
            center_point = (
                np.array([face_rgb.shape[1], face_rgb.shape[0]]) / 2.0
            )  # x, y (i.e. width, height)

            # rectifying center point
            center_point[1] = center_point[1] + center_point[1] * 0.15
            scale = (center_point[0] * 1.25) * 2 / 256
            # NOTE: center crop
            im = center_crop(
                face_rgb, center_point, scale, (self.img_size[0], self.img_size[1])
            )
            im = im.astype("float32")
            # NOTE: normalize, rgb order (shape is HWC order and C is rgb order)
            im = (im / 255.0 - self.mean) / self.std
            im = im.transpose([2, 0, 1])  # HWC->CHW
            im_tensor = np.expand_dims(im, 0)  # 1CHW

            # NOTE: inference
            net_out = self.ort_session.run(
                self.outputs, input_feed={self.input_name: im_tensor}
            )
            score_map = net_out[0]
            x_offset = net_out[1]
            y_offset = net_out[2]
            offset = [x_offset, y_offset]
            encoded_resolution = (score_map.shape[2], score_map.shape[3])  # h, w

            scores = score_map.reshape(5, -1)
            scores = scores.max(axis=1)

            # scores = scores.detach().numpy()
            # NOTE: post-process
            pts_img_coord = self.pp.decode_preds(
                score_map,
                [center_point],
                [scale],
                encoded_resolution,
                self.mask,
                offset,
            )

            # NOTE: points coordniates in frame
            pts_img_coord[:, :, 0] += face_crop_coord[0]
            pts_img_coord[:, :, 1] += face_crop_coord[1]
            pts_img_coord = pts_img_coord.squeeze(0)  # [n_landmarks, 2]
            face_landmarks.append(pts_img_coord)
            lmk_scores.append(scores)

        return face_landmarks, lmk_scores


def build_model(_file):
    return FaceLandmarker(_file)
