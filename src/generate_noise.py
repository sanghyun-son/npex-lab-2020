import glob
import numpy as np
import imageio
import tqdm

img_list = sorted(glob.glob('../dataset/DIV2K_sub/eval/target/*.png'))

for img_name in tqdm.tqdm(img_list, ncols=80):
    # img: np.uint8
    img = imageio.imread(img_name).astype(np.float)
    n = 20 * np.random.randn(*img.shape)    # np.float
    # Danger
    img_noise = img + n
    img_noise = img_noise.clip(min=0, max=255)
    img_noise = img_noise.round()
    img_noise = img_noise.astype(np.uint8)

    imageio.imwrite(img_name.replace('target', 'input'), img_noise)


