import os
import torch
import torch.utils.data as data
from .image_augmentation import *
from .gps_render import GPSDataRender, GPSImageRender

class ImageGPSDataset(data.Dataset):
    def __init__(self, args, image_list, sat_root="", mask_root="",
                 gps_root="", sat_type="png", mask_type="png", gps_type="data",
                 feature_embedding="", aug_mode="", randomize=True, aug_sampling_rate=None, aug_precision_rate=None):
        self.dataset_name = args.dataset_name
        self.image_list = image_list
        self.sat_root = sat_root
        self.mask_root = mask_root
        self.gps_root = gps_root
        self.sat_type = sat_type
        self.mask_type = mask_type
        self.randomize = randomize
        self.gps_type = gps_type
        self.delta = args.delta

        if gps_type == '':
            self.gps_render = None
        elif gps_type == 'image':
            self.gps_render = GPSImageRender(gps_root, 'png')
        elif gps_type == 'data':
            self.gps_render = GPSDataRender(args.quantity_render_type, gps_root, feature_embedding, aug_mode, aug_sampling_rate, aug_precision_rate)
        elif gps_type == 'npy':
            self.gps_render = GPSImageRender(gps_root, 'npy')


    def _read_image_and_mask(self, image_id):
        if self.sat_root != "":
            if self.gps_type == 'npy':
                img = np.load(os.path.join(self.sat_root, "{0}_sat.npy").format(image_id))
                img = np.squeeze(img) #(c,h,w)
            else:
                img = cv2.imread(os.path.join(self.sat_root, "{0}_sat.{1}").format(image_id, self.sat_type))
        else:
            img = None
        if self.mask_type == 'sdf':
            mask = np.load(os.path.join(self.mask_root, "{}_mask.npy").format(image_id))
            # mask = np.squeeze(mask)
        else:
            mask = cv2.imread(os.path.join(self.mask_root,  "{}_mask.png").format(image_id), cv2.IMREAD_GRAYSCALE)

        if mask is None: print("[WARN] empty mask: ", image_id)
        if self.gps_type == 'npy':
            mask = cv2.resize(mask, (img.shape[1], img.shape[2]), interpolation=cv2.INTER_NEAREST) #(h,w,c)
        return img, mask

    def _render_gps_to_image(self, image_id):
        ix, iy = image_id.split('_')
        if self.dataset_name == 'bj':#两个数据集的GPS数据命名方式不同
            x = int(iy)
            y = int(ix)
        elif self.dataset_name == 'sz':
            x = int(ix)
            y = int(iy)
        else:
            print("[ERROR] Unkown dataset name: ", self.dataset_name)
            exit(1)
        if self.gps_type == 'npy':
            gps_image = self.gps_render.render_npy(x, y)
        else:
            gps_image = self.gps_render.render(x, y)
        return gps_image

    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:

            if self.gps_type == "npy":
                img = np.concatenate([image1, image2], 0)
            else:
                img = np.concatenate([image1, image2], 2)

        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, sat, mask, gps_image, randomize=True):

        if randomize:
            if sat is not None:
                sat = randomHueSaturationValue(sat)
            img = self._concat_images(sat, gps_image)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        else:
            img = self._concat_images(sat, gps_image)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        try:
            if self.gps_type == "npy":
                img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
            else:
                img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)
        if self.mask_type == "png":
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
        return img, mask


    def __getitem__(self, index):
        image_id = self.image_list[index]
        if self.gps_render is not None:
            gps_img = self._render_gps_to_image(image_id)
        else:
            gps_img = None
        img, mask = self._read_image_and_mask(image_id)
        if self.mask_type == 'sdf':
            if self.delta == 0:
                self.delta = 1
            mask = np.clip(mask, -self.delta, self.delta) / self.delta
        #cv2.imwrite(f'dataset/GPS_image_direct/{image_id}_gps.png', gps_img) #保存GPS图像
        img, mask = self._data_augmentation(img, mask, gps_img, self.randomize)
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        return img, mask


    def __len__(self):
        return len(self.image_list)
