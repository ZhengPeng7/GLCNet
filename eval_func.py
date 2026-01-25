import os.path as osp
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from utils.km import run_kuhn_munkres
from utils.utils import write_json
from configs import config


CPU_COUNT = 8

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def _process_single_detection(args):
    """Process detection for a single image (for parallel execution)."""
    anno, det, det_thresh, iou_thresh, labeled_only = args

    gt_boxes = anno["boxes"]
    if labeled_only:
        inds = np.where(anno["pids"].ravel() != 5555)[0]
        if len(inds) == 0:
            return [], [], 0, 0
        gt_boxes = gt_boxes[inds]
    num_gt = gt_boxes.shape[0]

    if len(det) != 0:
        det = np.asarray(det)
        inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
        det = det[inds]
        num_det = det.shape[0]
    else:
        num_det = 0

    if num_det == 0:
        return [], [], 0, num_gt

    ious = np.zeros((num_gt, num_det), dtype=np.float32)
    for i in range(num_gt):
        for j in range(num_det):
            ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
    tfmat = ious >= iou_thresh

    for j in range(num_det):
        largest_ind = np.argmax(ious[:, j])
        for i in range(num_gt):
            if i != largest_ind:
                tfmat[i, j] = False

    for i in range(num_gt):
        largest_ind = np.argmax(ious[i, :])
        for j in range(num_det):
            if j != largest_ind:
                tfmat[i, j] = False

    y_true = [tfmat[:, j].any() for j in range(num_det)]
    y_score = [det[j, -1] for j in range(num_det)]
    count_tp = tfmat.sum()

    return y_true, y_score, count_tp, num_gt


def eval_detection(
    gallery_dataset, gallery_dets, det_thresh=0.5, iou_thresh=0.5, labeled_only=False,
    num_workers=CPU_COUNT,
):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image
    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    num_workers (int): number of parallel workers
    """
    assert len(gallery_dataset) == len(gallery_dets)
    annos = gallery_dataset.annotations

    # Prepare arguments for parallel processing
    args_list = [(anno, det, det_thresh, iou_thresh, labeled_only)
                 for anno, det in zip(annos, gallery_dets)]

    # Parallel processing
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(_process_single_detection, args_list)
    else:
        results = [_process_single_detection(args) for args in tqdm(args_list, desc="Detection eval", ncols=0)]

    # Aggregate results
    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for yt, ys, tp, gt in results:
        y_true.extend(yt)
        y_score.extend(ys)
        count_tp += tp
        count_gt += gt

    det_rate = count_tp * 1.0 / count_gt if count_gt > 0 else 0.0
    if len(y_true) > 0 and len(y_score) > 0:
        ap = average_precision_score(y_true, y_score) * det_rate
    else:
        ap = 0.0

    print("\n{} detection:".format("labeled only" if labeled_only else "all"))
    print("  recall = {:.2%}".format(det_rate))
    if not labeled_only:
        print("  ap = {:.2%}".format(ap))
    return det_rate, ap


def _process_single_query_protocol(args):
    """Process a single query for protocol-based datasets (CUHK-SYSU, MVN)."""
    (i, protoc_query, protoc_gallery, feat_q, query_dets_i, query_feats_i,
     name_to_det_feat, gallery_imgs, use_full_set, cbgm, k1, k2, topk,
     dataset_type, save_limit) = args

    y_true, y_score = [], []
    imgs, rois = [], []
    count_gt, count_tp = 0, 0

    query_imname = str(protoc_query["imname"][0, 0][0])
    # Use float64 for query_roi to avoid precision loss, and avoid in-place modification
    query_roi_raw = protoc_query["idlocate"][0, 0][0].astype(np.float64)
    query_roi = np.array([query_roi_raw[0], query_roi_raw[1],
                          query_roi_raw[0] + query_roi_raw[2],
                          query_roi_raw[1] + query_roi_raw[3]], dtype=np.float64)
    query_gt = []
    tested = set([query_imname])

    name2sim = {}
    name2gt = {}
    sims = []
    imgs_cbgm = []

    # Ensure query feature is float64 for numerical stability
    feat_q = feat_q.astype(np.float64)

    # 1. Go through the gallery samples defined by the protocol
    for item in protoc_gallery.squeeze():
        gallery_imname = str(item[0][0])
        # Parse gt based on dataset type - use float64 to avoid precision loss
        if dataset_type == 'cuhk':
            gt = item[1][0].astype(np.float64)
        else:  # mvn
            gt = item[1].ravel().astype(np.float64)
        count_gt += gt.size > 0

        if gallery_imname not in name_to_det_feat:
            continue
        det, feat_g = name_to_det_feat[gallery_imname]
        if det.shape[0] == 0:
            continue

        assert feat_g.size == np.prod(feat_g.shape[:2])
        feat_g = feat_g.reshape(feat_g.shape[:2]).astype(np.float64)
        sim = feat_g.dot(feat_q).ravel()

        if gallery_imname in name2sim:
            continue
        name2sim[gallery_imname] = sim
        name2gt[gallery_imname] = gt
        sims.extend(list(sim))
        imgs_cbgm.extend([gallery_imname] * len(sim))

    # 2. Go through remaining gallery images if using full set
    if use_full_set:
        for gallery_imname in gallery_imgs:
            if gallery_imname in tested:
                continue
            if gallery_imname not in name_to_det_feat:
                continue
            det, feat_g = name_to_det_feat[gallery_imname]
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2]).astype(np.float64)
            sim = feat_g.dot(feat_q).ravel()
            label = np.zeros(len(sim), dtype=np.int32)
            y_true.extend(list(label))
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))

    # 3. Apply CBGM if enabled
    if cbgm and query_dets_i is not None and query_feats_i is not None:
        sims = np.array(sims, dtype=np.float64)
        imgs_cbgm = np.array(imgs_cbgm)
        inds = np.argsort(sims)[-k1:]
        imgs_cbgm = set(imgs_cbgm[inds])
        for img in imgs_cbgm:
            sim = name2sim[img]
            det, feat_g = name_to_det_feat[img]
            qboxes = query_dets_i[:k2]
            qfeats = query_feats_i[:k2].astype(np.float64)
            assert np.abs(query_roi - qboxes[0][:4].astype(np.float64)).sum() <= 0.01, "query_roi must be the first one in pboxes"

            graph = []
            feat_g_f64 = feat_g.astype(np.float64)
            for indx_i, pfeat in enumerate(qfeats):
                for indx_j, gfeat in enumerate(feat_g_f64):
                    graph.append((indx_i, indx_j, float((pfeat * gfeat).sum())))
            km_res, max_val = run_kuhn_munkres(graph)

            for indx_i, indx_j, _ in km_res:
                if indx_i == 0:
                    sim[indx_j] = max_val
                    break

    # 4. Compute scores and labels
    for gallery_imname, sim in name2sim.items():
        gt_raw = name2gt[gallery_imname]
        det, feat_g = name_to_det_feat[gallery_imname]
        label = np.zeros(len(sim), dtype=np.int32)
        if gt_raw.size > 0:
            # gt_raw is [x1, y1, w, h] format - convert to [x1, y1, x2, y2] without in-place modification
            w, h = float(gt_raw[2]), float(gt_raw[3])
            gt = np.array([gt_raw[0], gt_raw[1], gt_raw[0] + gt_raw[2], gt_raw[1] + gt_raw[3]], dtype=np.float64)
            query_gt.append({"img": str(gallery_imname), "roi": list(map(float, list(gt)))})
            iou_thresh = min(0.5, (w * h) / ((w + 10.0) * (h + 10.0)))
            inds = np.argsort(sim)[::-1]
            sim = sim[inds]
            det = det[inds]
            for j, roi in enumerate(det[:, :4]):
                if _compute_iou(roi.astype(np.float64), gt) >= iou_thresh:
                    label[j] = 1
                    count_tp += 1
                    break
        y_true.extend(list(label))
        y_score.extend(list(sim))
        imgs.extend([gallery_imname] * len(sim))
        rois.extend(list(det))
        tested.add(gallery_imname)

    # 5. Compute AP for this query
    y_score = np.asarray(y_score, dtype=np.float64)
    y_true = np.asarray(y_true)
    assert count_tp <= count_gt
    recall_rate = count_tp * 1.0 / count_gt
    ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
    inds = np.argsort(y_score)[::-1]
    y_score = y_score[inds]
    y_true = y_true[inds]
    acc = [min(1, sum(y_true[:k])) for k in topk]

    # 6. Save result for JSON dump (only wrong results)
    new_entry = None
    if not int(y_true[0]) and i < save_limit:
        new_entry = {
            "query_img": str(query_imname),
            "query_roi": list(map(float, list(query_roi))),
            "query_gt": query_gt,
            "gallery": [],
        }
        topN = len(imgs)
        for k in range(topN):
            new_entry["gallery"].append({
                "img": str(imgs[inds[k]]),
                "roi": list(map(float, list(rois[inds[k]]))),
                "score": round(float(y_score[k]), 5),
                "correct": int(y_true[k]),
            })

    return ap, acc, new_entry


def eval_search_cuhk(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats,
    query_box_feats,
    query_dets,
    query_feats,
    k1=10,
    k2=3,
    det_thresh=0.5,
    cbgm=False,
    gallery_size=100,
    num_workers=CPU_COUNT,
):
    """
    gallery_dataset/query_dataset: an instance of BaseDataset
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    gallery_feat (list of ndarray): n_det x D features per image
    query_feat (list of ndarray): D dimensional features per query image
    det_thresh (float): filter out gallery detections whose scores below this
    gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                        -1 for using full set
    num_workers (int): number of parallel workers
    """
    assert len(gallery_dataset) == len(gallery_dets)
    assert len(gallery_dataset) == len(gallery_feats)
    assert len(query_dataset) == len(query_box_feats)

    use_full_set = gallery_size == -1
    fname = "TestG{}".format(gallery_size if not use_full_set else 50)
    protoc = loadmat(osp.join(gallery_dataset.root, "annotation/test/train_test", fname + ".mat"))
    protoc = protoc[fname].squeeze()

    # mapping from gallery image to (det, feat)
    annos = gallery_dataset.annotations
    name_to_det_feat = {}
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats):
        name = anno["img_name"]
        if len(det) != 0:
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

    topk = [1, 5, 10]
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(len(query_dataset)):
        feat_q = query_box_feats[i].ravel()
        query_dets_i = query_dets[i] if cbgm and len(query_dets) > i else None
        query_feats_i = query_feats[i] if cbgm and len(query_feats) > i else None
        args_list.append((
            i, protoc["Query"][i], protoc["Gallery"][i], feat_q, query_dets_i, query_feats_i,
            name_to_det_feat, gallery_dataset.imgs, use_full_set, cbgm, k1, k2, topk,
            'cuhk', 100000
        ))

    # Parallel processing
    print(f"CUHK search: {len(args_list)} queries...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(_process_single_query_protocol, args_list)
    else:
        results = [_process_single_query_protocol(args) for args in tqdm(args_list, desc="CUHK search", ncols=0)]

    # Aggregate results
    aps = []
    accs = []
    for ap, acc, new_entry in results:
        aps.append(ap)
        accs.append(acc)
        if new_entry is not None:
            ret["results"].append(new_entry)

    print("search ranking:")
    mAP = np.mean(aps)
    print("  mAP = {:.2%}".format(mAP))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print("  top-{:2d} = {:.2%}".format(k, accs[i]))

    write_json(ret, "vis/results-cuhk.json")

    ret["mAP"] = mAP
    ret["accs"] = accs
    return ret


def eval_search_mvn(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats,
    query_box_feats,
    query_dets,
    query_feats,
    k1=10,
    k2=3,
    det_thresh=0.5,
    cbgm=False,
    gallery_size=100,
    num_workers=CPU_COUNT,
):
    """
    gallery_dataset/query_dataset: an instance of BaseDataset
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    gallery_feat (list of ndarray): n_det x D features per image
    query_feat (list of ndarray): D dimensional features per query image
    det_thresh (float): filter out gallery detections whose scores below this
    gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                        -1 for using full set
    num_workers (int): number of parallel workers
    """
    assert len(gallery_dataset) == len(gallery_dets)
    assert len(gallery_dataset) == len(gallery_feats)
    assert len(query_dataset) == len(query_box_feats)

    use_full_set = gallery_size == -1
    fname = "TestG{}".format(gallery_size if not use_full_set else 2000)
    protoc = loadmat(osp.join(gallery_dataset.root, "annotation/test/train_test", fname + ".mat"))
    protoc = protoc[fname].squeeze()

    # mapping from gallery image to (det, feat)
    annos = gallery_dataset.annotations
    name_to_det_feat = {}
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats):
        name = anno["img_name"]
        if det != []:
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

    topk = [1, 5, 10]
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(len(query_dataset)):
        feat_q = query_box_feats[i].ravel()
        query_dets_i = query_dets[i] if cbgm and len(query_dets) > i else None
        query_feats_i = query_feats[i] if cbgm and len(query_feats) > i else None
        args_list.append((
            i, protoc["Query"][i], protoc["Gallery"][i], feat_q, query_dets_i, query_feats_i,
            name_to_det_feat, gallery_dataset.imgs, use_full_set, cbgm, k1, k2, topk,
            'mvn', 100000
        ))

    # Parallel processing
    print(f"MVN search: {len(args_list)} queries...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(_process_single_query_protocol, args_list)
    else:
        results = [_process_single_query_protocol(args) for args in tqdm(args_list, desc="MVN search", ncols=0)]

    # Aggregate results
    aps = []
    accs = []
    for ap, acc, new_entry in results:
        aps.append(ap)
        accs.append(acc)
        if new_entry is not None:
            ret["results"].append(new_entry)

    print("search ranking:")
    mAP = np.mean(aps)
    print("  mAP = {:.2%}".format(mAP))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print("  top-{:2d} = {:.2%}".format(k, accs[i]))

    write_json(ret, "vis/results-mvn-appN_{}-GS_{}.json".format(config.mvn_train_appN, config.mvn_gallery_size))

    ret["mAP"] = mAP
    ret["accs"] = accs
    return ret


def _process_single_query_prw(args):
    """Process a single query for PRW dataset (for parallel execution)."""
    (i, query_anno, feat_p, query_det, query_feat, annos, name_to_det_feat,
     ignore_cam_id, cbgm, k1, k2, topk) = args

    y_true, y_score = [], []
    imgs, rois = [], []
    count_gt, count_tp = 0, 0

    query_imname = query_anno["img_name"]
    query_roi = query_anno["boxes"]
    query_pid = query_anno["pids"]
    query_cam = query_anno["cam_id"]

    # Find all occurence of this query
    gallery_imgs = []
    for x in annos:
        if query_pid in x["pids"] and x["img_name"] != query_imname:
            gallery_imgs.append(x)
    query_gts = {}
    for item in gallery_imgs:
        roi = item["boxes"][item["pids"] == query_pid]
        query_gts[item["img_name"]] = list(map(float, roi)) if isinstance(roi, list) else list(map(float, roi.squeeze()))

    # Construct gallery set for this query
    if ignore_cam_id:
        gallery_imgs = [x for x in annos if x["img_name"] != query_imname]
    else:
        gallery_imgs = [x for x in annos if x["img_name"] != query_imname and x["cam_id"] != query_cam]

    name2sim = {}
    sims = []
    imgs_cbgm = []
    # Ensure query feature is float64 for numerical stability
    feat_p = feat_p.astype(np.float64)
    # Go through all gallery samples
    for item in gallery_imgs:
        gallery_imname = item["img_name"]
        count_gt += gallery_imname in query_gts
        if gallery_imname not in name_to_det_feat:
            continue
        det, feat_g = name_to_det_feat[gallery_imname]
        assert feat_g.size == np.prod(feat_g.shape[:2])
        feat_g = feat_g.reshape(feat_g.shape[:2]).astype(np.float64)
        sim = feat_g.dot(feat_p).ravel()

        if gallery_imname in name2sim:
            continue
        name2sim[gallery_imname] = sim
        sims.extend(list(sim))
        imgs_cbgm.extend([gallery_imname] * len(sim))

    if cbgm and query_det is not None and query_feat is not None:
        sims = np.array(sims, dtype=np.float64)
        imgs_cbgm = np.array(imgs_cbgm)
        inds = np.argsort(sims)[-k1:]
        imgs_cbgm = set(imgs_cbgm[inds])
        for img in imgs_cbgm:
            sim = name2sim[img]
            det, feat_g = name_to_det_feat[img]
            qboxes = query_det[:k2]
            qfeats = query_feat[:k2].astype(np.float64)
            assert np.abs(query_roi - qboxes[0][:4].astype(np.float64)).sum() <= 0.01

            graph = []
            feat_g_f64 = feat_g.astype(np.float64)
            for indx_i, pfeat in enumerate(qfeats):
                for indx_j, gfeat in enumerate(feat_g_f64):
                    graph.append((indx_i, indx_j, float((pfeat * gfeat).sum())))
            km_res, max_val = run_kuhn_munkres(graph)

            for indx_i, indx_j, _ in km_res:
                if indx_i == 0:
                    sim[indx_j] = max_val
                    break

    for gallery_imname, sim in name2sim.items():
        det, feat_g = name_to_det_feat[gallery_imname]
        label = np.zeros(len(sim), dtype=np.int32)
        if gallery_imname in query_gts:
            gt = query_gts[gallery_imname]
            w, h = gt[2] - gt[0], gt[3] - gt[1]
            iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
            inds = np.argsort(sim)[::-1]
            sim = sim[inds]
            det = det[inds]
            for j, roi in enumerate(det[:, :4]):
                if _compute_iou(roi, gt) >= iou_thresh:
                    label[j] = 1
                    count_tp += 1
                    break
        y_true.extend(list(label))
        y_score.extend(list(sim))
        imgs.extend([gallery_imname] * len(sim))
        rois.extend(list(det))

    # Compute AP for this query - use float64 for numerical stability
    y_score = np.asarray(y_score, dtype=np.float64)
    y_true = np.asarray(y_true)
    assert count_tp <= count_gt
    recall_rate = count_tp * 1.0 / count_gt
    ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
    inds = np.argsort(y_score)[::-1]
    y_score = y_score[inds]
    y_true = y_true[inds]
    acc = [min(1, sum(y_true[:k])) for k in topk]

    # Prepare result entry
    new_entry = None
    if i < 100:
        new_entry = {
            "query_img": str(query_imname),
            "query_roi": list(map(float, list(query_roi.squeeze()))),
            "query_gt": query_gts,
            "gallery": [],
        }
        topN = len(imgs)
        for k in range(min(topN, len(inds))):
            new_entry["gallery"].append({
                "img": str(imgs[inds[k]]),
                "roi": list(map(float, list(rois[inds[k]]))),
                "score": round(float(y_score[k]), 5),
                "correct": int(y_true[k]),
            })

    return ap, acc, new_entry


def eval_search_prw(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats,
    query_box_feats,
    query_dets,
    query_feats,
    k1=30,
    k2=4,
    det_thresh=0.5,
    cbgm=False,
    ignore_cam_id=True,
    num_workers=CPU_COUNT,
):
    """
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    gallery_feat (list of ndarray): n_det x D features per image
    query_feat (list of ndarray): D dimensional features per query image
    det_thresh (float): filter out gallery detections whose scores below this
    gallery_size (int): -1 for using full set
    ignore_cam_id (bool): Set to True acoording to CUHK-SYSU,
                        although it's a common practice to focus on cross-cam match only.
    num_workers (int): number of parallel workers
    """
    assert len(gallery_dataset) == len(gallery_dets)
    assert len(gallery_dataset) == len(gallery_feats)
    assert len(query_dataset) == len(query_box_feats)

    annos = gallery_dataset.annotations
    name_to_det_feat = {}
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats):
        name = anno["img_name"]
        scores = det[:, 4].ravel()
        inds = np.where(scores >= det_thresh)[0]
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds])

    topk = [1, 5, 10]
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(len(query_dataset)):
        query_anno = query_dataset.annotations[i]
        feat_p = query_box_feats[i].ravel()
        query_det = query_dets[i] if cbgm and len(query_dets) > i else None
        query_feat = query_feats[i] if cbgm and len(query_feats) > i else None
        args_list.append((
            i, query_anno, feat_p, query_det, query_feat, annos, name_to_det_feat,
            ignore_cam_id, cbgm, k1, k2, topk
        ))

    # Parallel processing
    print(f"PRW search: {len(args_list)} queries...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(_process_single_query_prw, args_list)
    else:
        results = [_process_single_query_prw(args) for args in tqdm(args_list, desc="PRW search", ncols=0)]

    # Aggregate results
    aps = []
    accs = []
    for ap, acc, new_entry in results:
        aps.append(ap)
        accs.append(acc)
        if new_entry is not None:
            ret["results"].append(new_entry)

    print("search ranking:")
    mAP = np.mean(aps)
    print("  mAP = {:.2%}".format(mAP))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print("  top-{:2d} = {:.2%}".format(k, accs[i]))
    write_json(ret, "vis/results-prw.json")

    ret["mAP"] = np.mean(aps)
    ret["accs"] = accs
    return ret
