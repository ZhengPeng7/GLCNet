import os.path as osp

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset

from config import ConfigMVN


class MVN(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "MVN"
        self.img_prefix = osp.join(root, "Image")
        self.gallery_size = ConfigMVN().gallery_size
        self.test_mat = 'TestG{}'.format(self.gallery_size)
        self.test_mat_path = osp.join(root, "annotation/test/train_test/{}.mat".format(self.test_mat))
        self.protoc = loadmat(self.test_mat_path)["{}".format(self.test_mat)].squeeze()
        self.train_mat = 'Train_app{}'.format(ConfigMVN().train_appN)
        super(MVN, self).__init__(root, transforms, split)

    def _load_queries(self):
        queries = []
        for item in self.protoc["Query"]:
            img_name = str(item["imname"][0, 0][0])
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([-100]),  # dummy pid
                }
            )
        return queries

    def _load_split_img_names_all(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        # the whole gallery images
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool_{}.mat".format(self.gallery_size)))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a.squeeze()) for a in gallery_imgs]
        if self.split == "gallery":
            return gallery_imgs
        # all images
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0].squeeze()) for a in all_imgs]
        # the whole training images = all images - all gallery images
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "gallery":
            gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool_{}.mat".format(self.gallery_size)))
            gallery_imgs = gallery_imgs["pool"].squeeze()
            gallery_imgs = [str(a.squeeze()) for a in gallery_imgs]
            return gallery_imgs
        # training images
        train_imgs = loadmat(self.test_mat_path.replace(self.test_mat, self.train_mat))['Train']
        training_imgs = []
        for train_img in train_imgs[:, 2]:
            for app in train_img:
                fn = app[0][0]
                training_imgs.append(fn)
        return training_imgs

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        # load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        unlabeled_pid = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:
            img_name = str(img_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
            boxes = boxes[valid_index]
            name_to_boxes[img_name] = boxes.astype(np.int32)
            name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return

        # assign a unique pid from 1 to N for each identity
        if self.split == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/{}.mat".format(self.train_mat)))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[2]
                for idx_sc, (img_name, box, _) in enumerate(scenes):
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
        else:
            for index, item in enumerate(self.protoc):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
            pids = name_to_pids[img_name]
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": boxes,
                    "pids": pids,
                }
            )
        return annotations
