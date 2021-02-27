import torch
import numpy as np
from PIL import Image
import collections

try:
    import accimage
except ImportError:
    accimage = None
import random
import scipy.ndimage as ndimage


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180

        image = ndimage.interpolation.rotate(
            image, angle1, reshape=self.reshape, order=self.order)
        depth = ndimage.interpolation.rotate(
            depth, angle1, reshape=self.reshape, order=self.order)

        image = Image.fromarray(image)
        depth = Image.fromarray(depth)

        return {'image': image, 'depth': depth}


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.changeScale(image, self.size)
        depth = self.changeScale(depth, self.size, Image.NEAREST)

        return {'image': image, 'depth': depth}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


class CenterCrop(object):
    def __init__(self, size_image, size_depth):
        self.size_image = size_image
        self.size_depth = size_depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.centerCrop(image, self.size_image)
        depth = self.centerCrop(depth, self.size_image)

        ow, oh = self.size_depth
        depth = depth.resize((ow, oh))

        return {'image': image, 'depth': depth}

    def centerCrop(self, image, size):
        w1, h1 = image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        image = self.to_tensor(image)
        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 10
        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth}


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)

        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if self.transforms is None:
            return {'image': image, 'depth': depth}
        order = torch.randperm(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)

        return {'image': image, 'depth': depth}


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, depth = sample['image'], sample['depth']

        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'depth': depth}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor


class Scale_iBims1(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, depth,edges,calib,mask_invalid,mask_transp,mask_wall, \
        mask_wall_paras,mask_table,mask_table_paras,mask_floor,mask_floor_paras=sample['image'], sample['depth'],sample['edges'], \
                                                                sample['calib'],sample['mask_invalid'], sample['mask_transp'], \
                                                                sample['mask_wall'],sample['mask_wall_paras'],sample['mask_table'], \
                                                               sample['mask_table_paras'],sample['mask_floor'],sample['mask_floor_paras']

        image = self.changeScale(image, self.size)
        depth = self.changeScale(depth, self.size, Image.NEAREST)
        edges = self.changeScale(edges, self.size, Image.NEAREST)
        #calib = self.changeScale(calib, self.size)
        mask_invalid = self.changeScale(mask_invalid, self.size, Image.NEAREST)
        mask_transp = self.changeScale(mask_transp, self.size, Image.NEAREST)
        mask_wall=self.changeScale(mask_wall, self.size, Image.NEAREST)
        mask_table=self.changeScale(mask_table, self.size, Image.NEAREST)
        mask_floor=self.changeScale(mask_floor, self.size, Image.NEAREST)

        return {'image': image, 'depth': depth, 'edges': edges, 'calib': calib,
                'mask_invalid': mask_invalid, 'mask_transp': mask_transp,
                "mask_wall": mask_wall, "mask_wall_paras": mask_wall_paras, "mask_table": mask_table,
                "mask_table_paras": mask_table_paras, "mask_floor": mask_floor, "mask_floor_paras": mask_floor_paras}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


class CenterCrop_iBims1(object):
    def __init__(self, size_image, size_depth):
        self.size_image = size_image
        self.size_depth = size_depth

    def __call__(self, sample):
        image, depth, edges, calib, mask_invalid, mask_transp, mask_wall, \
        mask_wall_paras, mask_table, mask_table_paras, mask_floor, mask_floor_paras=sample['image'], sample['depth'], \
                                                                                    sample['edges'], \
                                                                                    sample['calib'], sample[
                                                                                        'mask_invalid'], sample[
                                                                                        'mask_transp'], \
                                                                                    sample['mask_wall'], sample[
                                                                                        'mask_wall_paras'], sample[
                                                                                        'mask_table'], \
                                                                                    sample['mask_table_paras'], sample[
                                                                                        'mask_floor'], sample[
                                                                                        'mask_floor_paras']
        image = self.centerCrop(image, self.size_image)
        depth = self.centerCrop(depth, self.size_image)
        edges = self.centerCrop(edges, self.size_image)
        #calib = self.centerCrop(calib, self.size_image)
        mask_invalid = self.centerCrop(mask_invalid, self.size_image)
        mask_transp = self.centerCrop(mask_transp, self.size_image)
        mask_wall=self.centerCrop(mask_wall, self.size_image)
        mask_table=self.centerCrop(mask_table, self.size_image)
        mask_floor=self.centerCrop(mask_floor, self.size_image)

        ow, oh = self.size_depth
        depth = depth.resize((ow, oh))
        edges = edges.resize((ow, oh))
        mask_invalid = mask_invalid.resize((ow, oh))
        mask_transp = mask_transp.resize((ow, oh))
        mask_wall=mask_wall.resize((ow, oh))
        mask_table=mask_table.resize((ow, oh))
        mask_floor=mask_floor.resize((ow, oh))

        return {'image': image, 'depth': depth, 'edges': edges, 'calib': calib,
                'mask_invalid': mask_invalid, 'mask_transp': mask_transp,
                "mask_wall": mask_wall, "mask_wall_paras": mask_wall_paras, "mask_table": mask_table,
                "mask_table_paras": mask_table_paras, "mask_floor": mask_floor, "mask_floor_paras": mask_floor_paras}

    def centerCrop(self, image, size):

        w1, h1 = image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image

class ToTensor_iBims1(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth, edges, calib, mask_invalid, mask_transp, mask_wall, \
        mask_wall_paras, mask_table, mask_table_paras, mask_floor, mask_floor_paras=sample['image'], sample['depth'], \
                                                                                    sample['edges'], sample['calib'], \
                                                                                    sample['mask_invalid'], sample['mask_transp'], \
                                                                                    sample['mask_wall'], sample['mask_wall_paras'], \
                                                                                    sample['mask_table'],sample['mask_table_paras'], \
                                                                                    sample['mask_floor'], sample['mask_floor_paras']
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit

        image = self.to_tensor(image)
        depth = self.to_tensor(depth).float()
        edges=self.to_tensor(edges)
        calib=self.to_tensor(calib).float()
        mask_invalid=self.to_tensor(mask_invalid)
        mask_transp=self.to_tensor(mask_transp)
        mask_wall=self.to_tensor(mask_wall)
        mask_table=self.to_tensor(mask_table)
        mask_floor=self.to_tensor(mask_floor)
        mask_wall_paras=torch.from_numpy(mask_wall_paras)
        mask_table_paras=torch.from_numpy(mask_table_paras)
        mask_floor_paras=torch.from_numpy(mask_floor_paras)

        return {'image': image, 'depth': depth, 'edges': edges, 'calib': calib,
                'mask_invalid': mask_invalid, 'mask_transp': mask_transp,
                "mask_wall": mask_wall,"mask_wall_paras": mask_wall_paras, "mask_table": mask_table,
                "mask_table_paras": mask_table_paras,"mask_floor": mask_floor, "mask_floor_paras": mask_floor_paras}

    def to_tensor(self, pic):

        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int, copy=False))
        elif pic.mode == 'F':
            #print np.array(pic, np.uint8, copy=False)
            #img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = torch.from_numpy(np.array(pic, np.float64, copy=False))
        elif pic.mode=='1':
            img=torch.from_numpy(np.array(pic, boolen, copy=False))
        else:
            img = torch.from_numpy(np.array(pic, np.uint8, copy=False))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if pic.mode == 'RGB':
            return img.float()/255
        else:
            return img.float()

class Normalize_iBims1(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, depth, edges, calib, mask_invalid, mask_transp, mask_wall, \
        mask_wall_paras, mask_table, mask_table_paras, mask_floor, mask_floor_paras=sample['image'], sample['depth'], \
                                                                                    sample['edges'], sample['calib'], \
                                                                                    sample['mask_invalid'], sample[
                                                                                        'mask_transp'], \
                                                                                    sample['mask_wall'], sample[
                                                                                        'mask_wall_paras'], \
                                                                                    sample['mask_table'], sample[
                                                                                        'mask_table_paras'], \
                                                                                    sample['mask_floor'], sample[
                                                                                        'mask_floor_paras']

        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'depth': depth, 'edges': edges, 'calib': calib,
                'mask_invalid': mask_invalid, 'mask_transp': mask_transp,
                "mask_wall": mask_wall, "mask_wall_paras": mask_wall_paras, "mask_table": mask_table,
                "mask_table_paras": mask_table_paras, "mask_floor": mask_floor, "mask_floor_paras": mask_floor_paras}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor