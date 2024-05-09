import os
import numpy as np
import imageio
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune


from run_nerf_helpers import *
from run_nerf import *

from load_llff import load_llff_data
from torch.nn.utils import prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PruneNet():
    def layer_wise_pruning(self, model, layers, activations, percent):

        pruned_weights = []
        for i in range(len(layers)):
            cumulative_activations = torch.sum(activations[i], dim=0)
            sorted_indices = torch.argsort(cumulative_activations.flatten())
            mask = torch.ones(cumulative_activations.shape, dtype=torch.uint8)
            mask.flatten()[sorted_indices[:int(cumulative_activations.numel() * percent)]] = 0

            layers[i].weight.requires_grad = False
            layers[i].weight = nn.Parameter(layers[i].weight * mask.unsqueeze(1))
            pruned_weights.append(torch.count_nonzero(layers[i].weight))

        param_size = 0
        pruned_param_size = 0
        pruned_index = 0
        pattern = re.compile(r"pts_linears\..*\.weight")

        for name, param in model.named_parameters():
            if pattern.match(name):
                pruned_param_size += (pruned_weights[pruned_index]) * param.element_size()
                pruned_index += 1
                param_size += param.nelement() * param.element_size()
            else:
                param_size += param.nelement() * param.element_size()
                pruned_param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size)
        pruned_size_all_mb = (pruned_param_size + buffer_size)

        print('Model size: {:.3f}MB'.format(size_all_mb/2**20))
        print('Pruned model size: {:.3f}MB'.format(pruned_size_all_mb/2**20))

        return layers


def prune_nerf(percent_to_prune):
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    pruned_model = render_kwargs_train['network_fn']

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)


    N_rand = args.N_rand

    render_kwargs_train['network_fn'].save_activations = True
    for i in range(args.sample_size):
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        model = (render_kwargs_train['network_fn'])
        # print("Activations: ", len(model.activations)," ", len(model.pts_linears), "\n", model.activations)
    render_kwargs_train['network_fn'].save_activations = False
    # # Pruning the model
    dnn_pruning = PruneNet()
    model.pts_linears = dnn_pruning.layer_wise_pruning(model, model.pts_linears, model.activations, percent_to_prune)

    with torch.no_grad():
        # render_kwargs_test['network_fn'].save_activations = render_kwargs_train['network_fn'].save_activations = False
        print(render_kwargs_test['network_fn'].save_activations)
        render_kwargs_test['network_fn'] = render_kwargs_train['network_fn']
        rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_train)
    print('Done, saving', rgbs.shape, disps.shape)
    moviebase = os.path.join(args.basedir, args.expname, '{}_spiral_pruned_{}_'.format(args.expname, int(percent_to_prune*100)))
    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

    # # Calculating loss and PSNR over test set
    loss = 0.0
    psnr = 0.0
    with torch.no_grad():
        for i in i_train:
            # print(i)
            # img_i = np.random.choice(i_train)
            target = images[i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                #####  Core optimization loop  #####
                rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                        verbose=i < 10, retraw=True,
                                                        **render_kwargs_train)

                img_loss = img2mse(rgb, target_s)
                loss += img_loss
                psnr += mse2psnr(img_loss)

                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s)
                    loss = loss + img_loss0
                    psnr0 = mse2psnr(img_loss0)
    
    loss = loss / len(i_train)
    psnr = psnr / len(i_train)
    print(f"Over test set, \n loss: {loss} \n psnr : {psnr}")


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    plist = [0.0, 0.15, 0.3, 0.5, 0.85]
    for p in plist: 
        prune(p)
