import random
from PIL import Image
import glob
from rembg import remove
import tqdm


train_path  = glob.glob('./train/*')
background_path = glob.glob('./background/*')


for path in tqdm.tqdm(train_path):
    img = Image.open(path)
    img_name = './train_2/'+path.split('/')[-1]
    background = [Image.open(img_path).resize((400, 400)) for img_path in background_path]
    background_img = background[random.randint(0, len(background)-1)]
    rem_img_mask = remove(img, only_mask=True)
    background_img.paste(img, mask=rem_img_mask)
    background_img.save(img_name)
