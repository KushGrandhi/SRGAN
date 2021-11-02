from math import log10

import torch.utils.data
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
from srgan import Generator

model = Generator(2).eval()

if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('weights/2x/netG_epoch_2_49.pth'))

image = Image.open('/content/input/pic11.jpg')
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
image = image.cuda()

with torch.no_grad():
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('/content/result/pic11_.jpeg')