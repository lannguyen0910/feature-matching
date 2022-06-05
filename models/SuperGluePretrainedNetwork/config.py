resize = [-1, ]
resize_float = True
config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 10,
        "match_threshold": 0.2,
    }
}
