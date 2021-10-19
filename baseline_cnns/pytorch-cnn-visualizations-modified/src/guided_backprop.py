"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU, SiLU, GELU

import torch.nn.functional as F

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.forward_silu_inputs = []
        self.forward_gelu_inputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.update_silus()
        self.update_gelus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        if "VisionTransformer" in str(self.model.__class__):
            first_layer = self.model.patch_embed.proj
        else:
            print("First layer:", list(self.model._modules.items())[0])
            first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        relu_count = 0
        for pos, module in self.model.named_modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
                relu_count += 1
        print("Number of ReLUs:", relu_count)

    def update_silus(self):
        """
            Updates silu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def silu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            z = self.forward_silu_inputs.pop()
            modified_grad_out = torch.sigmoid(z) * (1 + z * (1 - torch.sigmoid(z))) * torch.clamp(grad_in[0], min=0.0)
            return (modified_grad_out,)

        def silu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_silu_inputs.append(ten_in[0])

        # Loop through layers, hook up SiLUs
        silu_count = 0
        for pos, module in self.model.named_modules():
            if isinstance(module, SiLU):
                module.register_backward_hook(silu_backward_hook_function)
                module.register_forward_hook(silu_forward_hook_function)
                silu_count += 1
        print("Number of SiLUs:", silu_count)

    def update_gelus(self):
        """
            Updates gelu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def gelu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            z = self.forward_gelu_inputs.pop()
            modified_grad_out = (0.5 * torch.tanh(0.0356774 * z**3 + 0.797885 * z) + (0.0535161 * z**3 + 0.398942 * z) / torch.cosh(0.0356774 * z**3 + 0.797885 * z)**2 + 0.5) * torch.clamp(grad_in[0], min=0.0)
            return (modified_grad_out,)

        def gelu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_gelu_inputs.append(ten_in[0])

        # Loop through layers, hook up GELUs
        gelu_count = 0
        for pos, module in self.model.named_modules():
            if isinstance(module, GELU):
                module.register_backward_hook(gelu_backward_hook_function)
                module.register_forward_hook(gelu_forward_hook_function)
                gelu_count += 1
        print("Number of GELUs:", gelu_count)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')
