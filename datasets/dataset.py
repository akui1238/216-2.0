# import os
# from PIL import Image
# import numpy as np
# from torch.utils.data import Dataset
#
#
# class NPY_datasets(Dataset):
#     def __init__(self, path_Data, config, train=True):
#         super(NPY_datasets, self).__init__()
#         self.path_Data = path_Data
#         self.train = train
#         self.transformer = config.train_transformer if train else config.test_transformer
#         self.data = self.load_data()
#
#     def load_data(self):
#         data = []
#         images_dir = self.path_Data + ('train/' if self.train else 'val/') + 'images/'
#         masks_dir = self.path_Data + ('train/' if self.train else 'val/') + 'masks/'
#
#         try:
#             # images_list = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
#             # masks_list = [f for f in os.listdir(masks_dir) if f.endswith('.gif')]
#
#             images_list = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
#             masks_list = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
#
#             print(f"Images found: {len(images_list)}")
#             print(f"Masks found: {len(masks_list)}")
#
#             if len(images_list) == 0 or len(masks_list) == 0:
#                 raise FileNotFoundError("No images or masks found in the directory!")
#
#             images_list = sorted(images_list)
#             masks_list = sorted(masks_list)
#
#             for img_name, msk_name in zip(images_list, masks_list):
#                 img_path = os.path.join(images_dir, img_name)
#                 msk_path = os.path.join(masks_dir, msk_name)
#                 data.append([img_path, msk_path])
#         except FileNotFoundError as e:
#             print(e)
#             # Handle the error or exit
#             exit(1)
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             # Handle other exceptions
#             exit(1)
#
#         return data
#
#     def __getitem__(self, indx):
#         img_path, msk_path = self.data[indx]
#         img = np.array(Image.open(img_path).convert('RGB'))
#         msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
#         img, msk = self.transformer((img, msk))
#         return img, msk
#
#     def __len__(self):
#         return len(self.data)


from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True
                 ,transforms=None):#+
        super(NPY_datasets, self)
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(path_Data + 'val/images/')
            masks_list = os.listdir(path_Data + 'val/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)



