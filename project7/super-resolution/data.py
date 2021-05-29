import glob
import shutil
import random
import cv2
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


class DIV2K:
    def __init__(self,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 images_dir='.div2k/images',
                 caches_dir='.div2k/caches'):

        self._ntire_2018 = True

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        _downgrades_a = ['bicubic', 'unknown']
        _downgrades_b = ['mild', 'difficult']

        if scale == 8 and downgrade != 'bicubic':
            raise ValueError(f'scale 8 only allowed for bicubic downgrade')

        if downgrade in _downgrades_b and scale != 4:
            raise ValueError(f'{downgrade} downgrade requires scale 4')

        if downgrade == 'bicubic' and scale == 8:
            self.downgrade = 'x8'
        elif downgrade in _downgrades_b:
            self.downgrade = downgrade
        else:
            self.downgrade = downgrade
            self._ntire_2018 = False

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        if not os.path.exists(self._hr_images_dir()):
            download_archive(self._hr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
        if not os.path.exists(self._lr_images_dir()):
            download_archive(self._lr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_HR.cache')

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.cache')

    def _hr_cache_index(self):
        return f'{self._hr_cache_file()}.index'

    def _lr_cache_index(self):
        return f'{self._lr_cache_file()}.index'

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

    def _lr_image_file(self, image_id):
        if not self._ntire_2018 or self.scale == 8:
            return f'{image_id:04}x{self.scale}.png'
        else:
            return f'{image_id:04}x{self.scale}{self.downgrade[0]}.png'

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, f'DIV2K_{self.subset}_HR')

    def _lr_images_dir(self):
        if self._ntire_2018:
            return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}')
        else:
            return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}')

    def _hr_images_archive(self):
        return f'DIV2K_{self.subset}_HR.zip'

    def _lr_images_archive(self):
        if self._ntire_2018:
            return f'DIV2K_{self.subset}_LR_{self.downgrade}.zip'
        else:
            return f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.zip'

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


# -----------------------------------------------------------
#  IO
# -----------------------------------------------------------


def download_archive(file, target_dir, extract=True):
    source_url = f'http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}'
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))




# -----------------------------------------------------------
#  Class Stanfords Dogs Dataset
# -----------------------------------------------------------


class STFDOGS20580:
    def __init__(self,
                 subset='train',
                 n_images=20580,
                 images_dir='.stfdogs20580/images',
                 caches_dir='.stfdogs20580/caches'):

        self.n_images = n_images         
        if n_images > 20580:
            raise ValueError("Le maximum est 20580")

        self.limit = int(n_images*80/100)

        if subset == 'train':
            self.image_ids = range(1, self.limit+1)
        elif subset == 'valid':
            self.image_ids = range(self.limit+1, self.n_images+1)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.downgrade = 'bicubic'
        self.scale = 4
        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())
        
        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())
        
        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f"STFDOGS20580_{self.subset}_HR.cache" )

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, f"STFDOGS20580_{self.subset}_LR_{self.downgrade}.cache" )

    def _hr_cache_index(self):
        return f"{self._hr_cache_file()}.index"

    def _lr_cache_index(self):
        return f"{self._lr_cache_file()}.index"


    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f"{image_id:05}.png") for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, f"{image_id:05}x{self.scale}.png") for image_id in self.image_ids]


    def _hr_images_dir(self):
        return os.path.join(self.images_dir, f"STFDOGS20580_{self.subset}_HR")

    def _lr_images_dir(self):
        return os.path.join(self.images_dir, f"STFDOGS20580_{self.subset}_LR_{self.downgrade}")

    def _stfdogs20580_images_dir(self):
        return os.path.join(self.images_dir, 'Images')

    def _download_stfdogs20580(self, filename='images.tar', target_dir='.stfdogs20580/images', extract=True):
        source_url = f'http://vision.stanford.edu/aditya86/ImageNetDogs/{filename}'
        target_dir = os.path.abspath(target_dir)
        tf.keras.utils.get_file(filename, source_url, cache_subdir=target_dir, extract=extract)
        os.remove(os.path.join(target_dir, filename))

    def _process_1(self):
        print("\nDéplacement des données dans dossier HR & shuffle & formatage")
        target_dir = os.path.join(self.images_dir, "Images")
        target_dir = os.path.abspath(target_dir)

        dst_dir = os.path.join(self.images_dir, f"STFDOGS20580_HR")
        dst_dir = os.path.abspath(dst_dir)

        if os.path.isdir(dst_dir) is False :
            os.mkdir(dst_dir)

        img_paths = glob.glob(f'{target_dir}\*\*.jpg') 
        random.shuffle(img_paths)

        img_paths = img_paths[0:self.n_images]

        progbar_1 = tf.keras.utils.Progbar(target=len(img_paths))
        for i, img_path in enumerate(img_paths,1):
            old_path = shutil.copy(img_path, dst_dir)
            new_path = f'{dst_dir}\\{i:05}.jpg'
            os.rename(old_path,new_path)
            progbar_1.add(1)
        
    def _process_2(self):
        print("\nDownsampling des données dans dossier LR")
        dst_dir = os.path.join(self.images_dir, f"STFDOGS20580_LR")
        dst_dir = os.path.abspath(dst_dir)

        target_dir = os.path.join(self.images_dir, f"STFDOGS20580_HR")
        target_dir = os.path.abspath(target_dir)

        if os.path.isdir(dst_dir) is False :
            os.mkdir(dst_dir)

        img_paths = glob.glob(f'{target_dir}\*.jpg')
        
        progbar_2 = tf.keras.utils.Progbar(target=len(img_paths))
        for i, img_path in enumerate(img_paths,1):
            img = cv2.imread(img_path)
            img = img.astype(np.uint8)
            cv2.imwrite(f'{target_dir}\\{i:05}.png',img)
            os.remove(img_path)

            scale = 4
            width = int(img.shape[1] * 1/scale)
            height = int(img.shape[0] * 1/scale)
            dim = (width, height)

            img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
            img_resized = img_resized.astype(np.uint8)
            cv2.imwrite(f'{dst_dir}\\{i:05}x{scale}.png',img_resized)
            progbar_2.add(1)

    def _process_3(self):
        print("\nSéparation des données HR (train,valid): ")

        train_dir_hr = os.path.join(self.images_dir, f"STFDOGS20580_train_HR")
        valid_dir_hr = os.path.join(self.images_dir, f"STFDOGS20580_valid_HR")

        os.mkdir(train_dir_hr)
        os.mkdir(valid_dir_hr)
        
        target_dir= os.path.join(self.images_dir, f"STFDOGS20580_HR")
        target_dir = os.path.abspath(target_dir)

        img_paths = glob.glob(f'{target_dir}\*.png')
        
        progbar_3 = tf.keras.utils.Progbar(target=len(img_paths))
        for img in img_paths[0:self.limit]:
            shutil.move(img, train_dir_hr)
            progbar_3.add(1)
        for img in img_paths[self.limit:self.n_images]:
            shutil.move(img, valid_dir_hr)
            progbar_3.add(1)
        os.rmdir(target_dir)

    def _process_4(self):
        print("\nSéparation des données LR (train,valid): ")

        train_dir_lr_bicubic = os.path.join(self.images_dir, f"STFDOGS20580_train_LR_bicubic")
        valid_dir_lr_bicubic = os.path.join(self.images_dir, f"STFDOGS20580_valid_LR_bicubic")

        os.mkdir(train_dir_lr_bicubic)
        os.mkdir(valid_dir_lr_bicubic)

        target_dir = os.path.join(self.images_dir, f"STFDOGS20580_LR")
        target_dir = os.path.abspath(target_dir)

        img_paths = glob.glob(f'{target_dir}\*.png')  
        
        progbar_4 = tf.keras.utils.Progbar(target=len(img_paths))
        for img in img_paths[0:self.limit]:
            shutil.move(img, train_dir_lr_bicubic)
            progbar_4.add(1)
        for img in img_paths[self.limit:self.n_images]:
            shutil.move(img, valid_dir_lr_bicubic)
            progbar_4.add(1)
        os.rmdir(target_dir)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):

        if not ( os.path.exists(self._stfdogs20580_images_dir())):
            self._download_stfdogs20580()
        if not ( os.path.exists(self._lr_images_dir()) and os.path.exists(self._hr_images_dir()) ) :
            self._process_1()
            self._process_2()
            self._process_3()
            self._process_4()
          
        ds_1 = self.lr_dataset()
        ds_2 = self.hr_dataset()
        ds = tf.data.Dataset.zip((ds_1, ds_2))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def remove_data(self, cache=True, images_archive=True, images_preprocessed=True):
        
        if os.path.exists(self.images_dir):

            images_archive_dir = os.path.join(self.images_dir, "Images")
            train_dir_lr_bicubic = os.path.join(self.images_dir, f"STFDOGS20580_train_LR_bicubic")
            valid_dir_lr_bicubic = os.path.join(self.images_dir, f"STFDOGS20580_valid_LR_bicubic")
            train_dir_hr = os.path.join(self.images_dir, f"STFDOGS20580_train_HR")
            valid_dir_hr = os.path.join(self.images_dir, f"STFDOGS20580_valid_HR")

            if os.path.exists(images_archive_dir) and images_archive:
                shutil.rmtree(os.path.abspath(images_archive_dir))
            if os.path.exists(train_dir_lr_bicubic) and images_preprocessed:
                shutil.rmtree(os.path.abspath(train_dir_lr_bicubic))
            if os.path.exists(valid_dir_lr_bicubic) and images_preprocessed:
                shutil.rmtree(os.path.abspath(valid_dir_lr_bicubic))
            if os.path.exists(train_dir_hr) and images_preprocessed:
                shutil.rmtree(os.path.abspath(train_dir_hr))
            if os.path.exists(valid_dir_hr) and images_preprocessed:
                shutil.rmtree(os.path.abspath(valid_dir_hr))

        if os.path.exists(self.caches_dir) and cache:
            shutil.rmtree(os.path.abspath(self.caches_dir))
            os.makedirs(self.caches_dir, exist_ok=True)



    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.io.decode_png(x, channels=3,dtype=tf.uint8), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f"\nCréation du cache : {cache_file} ...")
        for _ in ds: pass





