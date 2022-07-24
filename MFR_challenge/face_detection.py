from __future__ import division
import numpy as np
import cv2
import onnxruntime
from itertools import product


def _get_prior_box(original_size, steps, min_sizes, clip, loc_preds):
    imh, imw = original_size
    mean = []

    for k in range(len(loc_preds)):
        feath = loc_preds[k].shape[1]
        featw = loc_preds[k].shape[2]

        f_kw = imw / steps[k]
        f_kh = imh / steps[k]

        for i, j in product(range(feath), range(featw)):
            # get center point which is normalized by the feature-map size
            cx = (j + 0.5) / f_kw
            cy = (i + 0.5) / f_kh

            # get anchor-size normalized by original input-size
            s_kw = min_sizes[k] / imw
            s_kh = min_sizes[k] / imh

            mean += [cx, cy, s_kw, s_kh]

    # output = torch.Tensor(mean).view(-1, 4)
    output = np.array(mean, dtype=loc_preds[0].dtype).reshape(-1, 4)

    if clip:
        output = np.clip(0, 1)

    return output


class PostProcessor:
    def __init__(self, cfg=None):
        super(PostProcessor, self).__init__()
        if cfg is not None:
            self.num_classes = cfg.NUM_CLASSES
            self.top_k = cfg.TOP_K
            self.nms_thresh = cfg.NMS_THRESH
            self.conf_thresh = cfg.CONF_THRESH
            self.variance = cfg.VARIANCE
            self.nms_top_k = cfg.NMS_TOP_K
        else:
            self.num_classes = 2
            self.top_k = 750
            self.nms_thresh = 0.3
            self.conf_thresh = 0.3
            self.variance = [0.1, 0.2]
            self.nms_top_k = 1000
        self.batch_size = 1  # NOTE: currently support only for 1 batch
        # NOTE: only sinlge image in batch is allowed
        assert self.batch_size == 1

    def decode(self, loc, priors, variances):
        # NOTE: decode offset regression in the training time
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        # NOTE decode center-x, center-y into x0, y0
        boxes[:, :2] -= boxes[:, 2:] / 2
        # NOTE get x1, y1 by adding offset value i.e. x0 and y0
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def nms(self, dets, scores):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1][
            : self.nms_top_k
        ]  # sort by descending order of confidence score

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep, len(keep)

    def forward(self, loc_data, conf_data, prior_data):
        # get batch_size
        num = self.batch_size
        # get # of prior boxes
        num_priors = prior_data.shape[0]

        conf_preds = conf_data.reshape(num, num_priors, self.num_classes).transpose(
            [0, 2, 1]
        )

        decoded_boxes = self.decode(loc_data.reshape(-1, 4), prior_data, self.variance)
        decoded_boxes = decoded_boxes.reshape(num, num_priors, 4)

        output = np.zeros(
            (num, self.num_classes, self.top_k, 5), dtype=loc_data.dtype
        )  # final output placeholder

        # NOTE: for each sample (in a batch)
        i = 0  # NOTE: only for single image
        boxes = decoded_boxes[i].copy()
        conf_scores = conf_preds[i].copy()

        count = 0
        # NOTE: do not care in case cl is 0 (due to zero is background-idx)
        for cl in range(1, self.num_classes):
            c_mask = conf_scores[cl] > self.conf_thresh
            scores = conf_scores[cl][c_mask]

            # NOTE: if all boxes are below conf_thresh
            if np.sum(c_mask) == 0:
                continue

            # NOTE: make mask based on confidence mask which has same shape with boxes.
            c_mask_unsqueezed = np.expand_dims(c_mask, 1)
            l_mask = np.tile(c_mask_unsqueezed, (1, boxes.shape[1]))
            boxes_ = boxes[l_mask].reshape(-1, 4)

            ids, count = self.nms(boxes_, scores)
            count = count if count < self.top_k else self.top_k

            output[i, cl, :count] = np.concatenate(
                (np.expand_dims(scores[ids[:count]], 1), boxes_[ids[:count]]), 1
            )
        return output[:, :, :count, :]


class FaceDetector:
    def __init__(self, onnx_file, img_size):
        self.onnx_file = onnx_file
        self.img_size = img_size

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
                "det intra_op_num_threads {} inter_op_num_threads {}".format(
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

        # anchor-box setups
        self.anchor_strides = [4, 8, 16, 32, 64, 128]
        self.anchor_sizes = [16, 32, 64, 128, 256, 512]
        self.anchor_clip = False

        # post processing setups
        self.pp = PostProcessor()

    def detect(self, img, scale=1.0):
        # NOTE: img is bgr order
        proposals_list = []
        scores_list = []
        landmarks_list = []
        if scale == 1.0:
            im = img
        else:
            im = cv2.resize(
                img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
            )

        im = im.astype("float32")
        # NOTE: normalize, bgr order (shape is HWC order and C is bgr order)
        im -= np.array([104.0, 117.0, 123.0])[np.newaxis, np.newaxis, :].astype(
            "float32"
        )

        im_info = [im.shape[0], im.shape[1]]
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):  # NOTE: BGR2RGB
            im_tensor[0, i, :, :] = im[:, :, 2 - i]

        im_tensor = im_tensor.astype(np.float32)
        net_out = self.ort_session.run(
            self.outputs, input_feed={self.input_name: im_tensor}
        )
        loc_data = net_out[0]
        conf_data = net_out[1]
        featuremap_size_ref_data = net_out[2:]
        prior_data = _get_prior_box(
            im_info,
            self.anchor_strides,
            self.anchor_sizes,
            self.anchor_clip,
            featuremap_size_ref_data,
        )

        # NOTE: post processing (e.g. nms, etc)
        detections = self.pp.forward(loc_data, conf_data, prior_data)
        origin_scale = [
            im.shape[1],
            im.shape[0],
            im.shape[1],
            im.shape[0],
        ]  # w, h, w, h

        proposals_list = []
        for j in range(len(detections[0, 1, :, 0])):
            score = detections[0, 1, j, 0]
            pt = detections[0, 1, j, 1:] * origin_scale
            xmin = pt[0]
            ymin = pt[1]
            xmax = pt[2]
            ymax = pt[3]
            proposals_list.append([xmin, ymin, xmax, ymax])

        if len(proposals_list) == 0:
            return np.zeros((0, 5))

        proposals = np.vstack(proposals_list)
        return proposals


def build_model(_file, img_size):
    return FaceDetector(_file, img_size)
