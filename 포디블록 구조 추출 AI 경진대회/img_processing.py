import random
from PIL import Image
import glob
from rembg import remove

test_path  = glob.glob('open/test/*')
train_path  = glob.glob('open/train/*')
background_path = glob.glob('background/*')
background = [Image.open(img_path).resize((400,400)) for img_path in background_path]

for img_path in [train_path, test_path]:
    for path in train_path:
        img = Image.open(path)
        img_name = './'+path.split('/')[1]+path.split('/')[2]
        background_img = background[random.randint(0, len(background)-1)]
        rem_img_mask = remove(img,only_mask = True)
        background_img.paste(img,mask = rem_img_mask)
        background_img.save(img_name)
