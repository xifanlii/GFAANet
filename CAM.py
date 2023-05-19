import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model.Backbone.MetaFormer import MetaFG_2


class ReshapeTransform:
    def __init__(self, model):
        input_size = 224 #model.patch_embed.img_size
        patch_size = 7 #model.patch_embed.patch_size
        self.h = 7 #input_size[0] // patch_size[0]
        self.w = 7 #input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():

    #加载模型
    model = MetaFG_2()
    weights_path = r"D:\WorkSpace\PyCharm Workspace\Fine_Grained\net_pth\Meta_5_224 _FGVC.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    #由于最终分类是在最后一个关注块中计算的类令牌上进行的，
    #输出将不受最后一层中的14x14通道的影响。
    #输出相对于它们的梯度将为0！
    #我们应该在最后一个关注块之前选择任何层。

    target_layers = [model.stage_4[-2].norm1]

    #数据集设置
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    # load image
    img_path = r"D:\WorkSpace\PyCharm Workspace\Fine_Grained\MyDataSet\FGVC-Aircraft\test\777-300\1179713.jpg"

    #加载图片
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 20  # tabby, tabby cat

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()