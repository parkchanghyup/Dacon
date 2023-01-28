import random
from PIL import Image
import glob
from rembg import remove
import tqdm


test_path  = glob.glob('/Users/harvey/Downloads/open/test/*')
train_path  = glob.glob('/Users/harvey/Downloads/open/train/*')
test_path.sort()
train_path.sort()
background_path = glob.glob('/Users/harvey/Downloads/background/*')




for idx,  path in tqdm.tqdm(enumerate(test_path)):
    img = Image.open(path)
    img_name = './test_2/'+path.split('/')[-1]
    background = [Image.open(img_path).resize((400, 400)) for img_path in background_path]
    background_img = background[idx%9]
    rem_img_mask = remove(img, only_mask = True)
    background_img.paste(img, mask = rem_img_mask)
    background_img.save(img_name)

for idx,  path in tqdm.tqdm(enumerate(train_path)):
    img = Image.open(path)
    img_name = './train_2/'+path.split('/')[-1]
    background = [Image.open(img_path).resize((400, 400)) for img_path in background_path]
    background_img = background[idx%9]
    rem_img_mask = remove(img, only_mask = True)
    background_img.paste(img, mask = rem_img_mask)
    background_img.save(img_name)
