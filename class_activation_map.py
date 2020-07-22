import torch
from skimage.transform import resize
import matplotlib.pyplot as plt
import wandb
from torch.autograd import Variable


class ClassActivationMap:
    def __init__(self, model):
        self.gradient = []

        self.h = model.last_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output = args[2]
        self.gradient.append(grad_output[0])

    def get_gradient(self, idx):
        return self.gradient[idx]

    def remove_hook(self):
        self.h.remove()

    def normalize_cam(self, x):
        x = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
        x[x < torch.max(x)] = -1
        return x

    def visualize(self, cam_img, img_var, wandb, data):
        x = img_var[0, :, :].cpu().data.numpy()
        cam_img = resize(cam_img.cpu().data.numpy(), output_shape=x.shape)
        plt.subplot(1, 3, 1)
        plt.imshow(cam_img)

        plt.subplot(1, 3, 2)
        plt.imshow(x, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(x + cam_img)
        plt.show()
        print(cam_img.shape)
        print(x.shape)

        wandb.log({"chart": plt})

    def get_cam(self, idx):
        grad = self.get_gradient(idx)
        # alpha = torch.sum(grad,dim=3,keepdim=True)
        # alpha = torch.sum(alpha,dim=2,keepdim=True)
        alpha = grad

        cam = alpha[idx] * grad[idx]
        cam = torch.sum(cam, dim=0)
        cam = self.normalize_cam(cam)

        self.remove_hook()
        return cam


def show_class_activation_map(model, valid_loader, wandb, upload_number):
    cam = ClassActivationMap(model)

    for b_idx, data in enumerate(valid_loader):
        x = Variable(data["image"]).cuda()
        y_ = Variable(data["targets"]).cuda()
        output, _ = model.forward(x, y_)
        for j in range(upload_number):
            model.zero_grad()
            label = y_[j]
            output[j, label].backward(retain_graph=True)
            out = cam.get_cam(j)
            cam.visualize(out, x[j], wandb, data)

        break
