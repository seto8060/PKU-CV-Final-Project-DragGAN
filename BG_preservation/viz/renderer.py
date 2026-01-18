# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                print(data[key].init_args)
                print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'G_ema')
        self.G = G
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Generate random latents.
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        self.w0 = w.detach().clone()
        self.w_plus = w_plus
        if w_plus:
            self.w = w.detach()
        else:
            self.w = w[:, 0, :].detach()
        self.w.requires_grad = True
        self.w_optim = torch.optim.Adam([self.w], lr=lr)

        self.feat_refs = None
        self.points0_pt = None
        
        # Feature blending variables
        self.drag_step = 0
        self.blend_interval = 50  # N: interval for feature blending
        self.reproject_steps = 0  # m: steps for re-projection
        self.reproject_lr = 0.001  # Learning rate for re-projection (can be adjusted)
        self.feature_hooks = []
        self.feat0_all_layers = None
        self.enable_feature_blending = False
        self.save_blend_images = True
        self.blend_save_dir = './blend_visualization'
        self.mask_blur_sigma = 5.0  # Gaussian blur sigma for mask smoothing

    def update_lr(self, lr):

        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

    def _register_feature_blending_hooks(self, mask, feature_idx=5):
        """
        Register hooks on synthesis network layers for feature blending.
        Args:
            mask: Mask tensor (1 for fixed area, 0 for flexible area)
            feature_idx: Index of the feature layer to use for blending
        """
        # Remove existing hooks
        self._remove_feature_blending_hooks()
        
        if mask is None or self.feat0_all_layers is None or feature_idx >= len(self.feat0_all_layers):
            return
        
        synthesis_net = self.G.synthesis
        feat0_layer = self.feat0_all_layers[feature_idx].detach()
        
        # Find the target layer (StyleGAN3: layer_names, StyleGAN2: block_resolutions)
        layer_to_hook = None
        if hasattr(synthesis_net, 'layer_names') and feature_idx < len(synthesis_net.layer_names):
            layer_to_hook = getattr(synthesis_net, synthesis_net.layer_names[feature_idx], None)
        elif hasattr(synthesis_net, 'block_resolutions') and feature_idx < len(synthesis_net.block_resolutions):
            res = synthesis_net.block_resolutions[feature_idx]
            layer_to_hook = getattr(synthesis_net, f'b{res}', None)
        
        if layer_to_hook is None:
            print(f"Warning: Could not find layer for feature_idx {feature_idx}")
            return
        
        # Hook to the specific layer - modifies feature before it's used by subsequent layers
        def layer_hook(module, input, output):
            if not self.enable_feature_blending:
                return output
            
            # Handle different output types: StyleGAN3 (tensor) vs StyleGAN2 (tuple)
            feat_current = output[0] if isinstance(output, tuple) else output
            if not isinstance(feat_current, torch.Tensor):
                return output
            
            # Resize mask and initial feature to match current feature dimensions
            B, C, H, W = feat_current.shape
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float().to(self._device), 
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            
            # Apply Gaussian blur to mask for smooth blending
            if self.mask_blur_sigma > 0:
                kernel_size = int(2 * self.mask_blur_sigma * 2 + 1)
                kernel_size = kernel_size + (1 - kernel_size % 2)  # Ensure odd
                # Create 1D Gaussian kernel
                x = torch.arange(kernel_size, dtype=torch.float32, device=self._device) - kernel_size // 2
                gaussian_1d = torch.exp(-(x ** 2) / (2 * self.mask_blur_sigma ** 2))
                gaussian_1d = gaussian_1d / gaussian_1d.sum()
                # Apply separable 1D convolution (more efficient than 2D)
                mask_2d = mask_resized.unsqueeze(0).unsqueeze(0)
                mask_blurred = F.conv2d(mask_2d, gaussian_1d[None, None, :, None], padding=(0, kernel_size // 2))
                mask_blurred = F.conv2d(mask_blurred, gaussian_1d[None, None, None, :], padding=(kernel_size // 2, 0))
                mask_resized = mask_blurred.squeeze(0).squeeze(0).clamp(0, 1)
            
            feat0_resized = F.interpolate(feat0_layer.to(self._device), size=(H, W), mode='bilinear', align_corners=False)
            
            # Blend: mask=1 uses initial feature, mask=0 uses current feature
            mask_expanded = mask_resized.unsqueeze(0).unsqueeze(0)
            blended = feat_current * (1 - mask_expanded) + feat0_resized * mask_expanded
            
            return (blended, output[1]) if isinstance(output, tuple) else blended
        
        self.feature_hooks.append(layer_to_hook.register_forward_hook(layer_hook))

    def _remove_feature_blending_hooks(self):
        """Remove all registered feature blending hooks."""
        for hook in self.feature_hooks:
            hook.remove()
        self.feature_hooks = []

    def _save_blend_visualization(self, img_before, img_after, step):
        """Save before/after blending visualization images."""
        try:
            os.makedirs(self.blend_save_dir, exist_ok=True)
            
            def tensor_to_image(img_tensor):
                img_tensor = img_tensor.detach().cpu()
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2.0
                img_array = (img_tensor.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                return Image.fromarray(img_array)
            
            for img, suffix in [(img_before, 'before'), (img_after, 'after')]:
                path = os.path.join(self.blend_save_dir, f'step_{step:04d}_{suffix}_blend.png')
                tensor_to_image(img).save(path)
            
            print(f'[Feature Blending] Saved images at step {step}')
        except Exception as e:
            print(f'[Feature Blending] Failed to save images: {e}')

    def _save_reproject_visualization(self, img_before, img_after, step):
        """Save before/after re-projection visualization images."""
        try:
            os.makedirs(self.blend_save_dir, exist_ok=True)
            
            def tensor_to_image(img_tensor):
                img_tensor = img_tensor.detach().cpu()
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2.0
                img_array = (img_tensor.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                return Image.fromarray(img_array)
            
            for img, suffix in [(img_before, 'before'), (img_after, 'after')]:
                path = os.path.join(self.blend_save_dir, f'step_{step:04d}_{suffix}_reproject.png')
                tensor_to_image(img).save(path)
            
            print(f'[Re-projection] Saved images at step {step}')
        except Exception as e:
            print(f'[Re-projection] Failed to save images: {e}')

    def _save_blending_three_images(self, img_before_blend, img_after_blend, img_after_reproject, step):
        """Save three images: before blending, after blending (re-projection target), after re-projection."""
        try:
            os.makedirs(self.blend_save_dir, exist_ok=True)
            
            def tensor_to_image(img_tensor):
                img_tensor = img_tensor.detach().cpu()
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2.0
                img_array = (img_tensor.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                return Image.fromarray(img_array)
            
            # Save three images with descriptive names
            images_to_save = [
                (img_before_blend, 'before_blend'),
                (img_after_blend, 'after_blend'),
                (img_after_reproject, 'after_reproject')
            ]
            
            for img, suffix in images_to_save:
                path = os.path.join(self.blend_save_dir, f'step_{step:04d}_{suffix}.png')
                tensor_to_image(img).save(path)
            
            print(f'[Blending] Saved three images at step {step}: before_blend, after_blend, after_reproject')
        except Exception as e:
            print(f'[Blending] Failed to save images: {e}')

    def _reproject_w(self, G, label, feat_blended_target, feature_idx, trunc_psi, noise_mode):
        """
        Re-project blended features back to W+ space.
        Performs m steps of W+ optimization to align current features with blended features.
        
        Args:
            G: Generator network
            label: Conditioning label
            feat_blended_target: Target blended feature (detached)
            feature_idx: Index of feature layer to align
            trunc_psi: Truncation psi
            noise_mode: Noise mode
        
        Returns:
            Updated ws tensor (constructed from optimized self.w)
        """
        # Create temporary optimizer with re-projection learning rate
        reproject_optim = torch.optim.Adam([self.w], lr=self.reproject_lr)
        
        print(f'[Re-projection] Starting {self.reproject_steps} steps of re-projection...')
        
        for reproj_step in range(self.reproject_steps):
            # Build ws from current self.w (same logic as in _render_drag_impl)
            ws_temp = self.w
            if ws_temp.dim() == 2:
                ws_temp = ws_temp.unsqueeze(1).repeat(1, 6, 1)
            ws_temp = torch.cat([ws_temp[:, :6, :], self.w0[:, 6:, :]], dim=1)
            
            # Forward pass to get current features
            _, feat_current = G(ws_temp, label, truncation_psi=trunc_psi, noise_mode=noise_mode, 
                               input_is_w=True, return_feature=True)
            
            # Compute feature loss (L1 loss between current and target features)
            feat_current_layer = feat_current[feature_idx]
            loss_reproject = F.l1_loss(feat_current_layer, feat_blended_target)
            
            # Optimize
            reproject_optim.zero_grad()
            loss_reproject.backward()
            reproject_optim.step()
            
            if (reproj_step + 1) % max(1, self.reproject_steps // 5) == 0 or reproj_step == 0:
                print(f'[Re-projection] Step {reproj_step + 1}/{self.reproject_steps}, Loss: {loss_reproject.item():.6f}')
        
        # Clean up temporary optimizer
        del reproject_optim
        
        # Build final ws from optimized self.w
        ws_final = self.w
        if ws_final.dim() == 2:
            ws_final = ws_final.unsqueeze(1).repeat(1, 6, 1)
        ws_final = torch.cat([ws_final[:, :6, :], self.w0[:, 6:, :]], dim=1)
        
        print(f'[Re-projection] Completed {self.reproject_steps} steps')
        
        return ws_final

    def _render_drag_impl(self, res,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        blend_interval  = None,  # N: interval for feature blending (None = use default)
        reproject_steps = None,  # m: steps for re-projection after blending (None = use default)
        **kwargs
    ):
        G = self.G
        ws = self.w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        # Update blend_interval and reproject_steps if provided
        if blend_interval is not None:
            self.blend_interval = blend_interval
        if reproject_steps is not None:
            self.reproject_steps = reproject_steps
            
        if reset:
            self.feat_refs = None
            self.points0_pt = None
            self.drag_step = 0
            self.feat0_all_layers = None
            self._remove_feature_blending_hooks()
        self.points = points

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        
        # Save initial features on first drag step or reset
        if is_drag and (self.feat0_all_layers is None or reset):
            with torch.no_grad():
                _, feat0 = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
                self.feat0_all_layers = [f.detach().clone() for f in feat0]
        
        # Enable feature blending at specified intervals
        should_blend = (is_drag and mask is not None and self.feat0_all_layers is not None and 
                       self.drag_step > 0 and self.drag_step % self.blend_interval == 0)
        img_before_blend = None
        
        if should_blend:
            # Save image before blending
            if self.save_blend_images:
                with torch.no_grad():
                    img_before, _ = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, 
                                     input_is_w=True, return_feature=True)
                    img_before_blend = img_before[0].clone()
            self.enable_feature_blending = True
            self._register_feature_blending_hooks(mask, feature_idx)
        else:
            self.enable_feature_blending = False
        
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

        # Re-projection after blending: align W+ with blended features
        if should_blend and is_drag:
            # Save blended feature as target
            feat_blended_target = feat[feature_idx].detach().clone()
            
            # Save image after blending (this is the re-projection target image)
            img_after_blend = None
            if self.save_blend_images:
                img_after_blend = img[0].clone()
            
            # Disable feature blending hooks before re-projection
            self.enable_feature_blending = False
            self._remove_feature_blending_hooks()
            
            # Perform re-projection: optimize W+ to match blended features
            ws = self._reproject_w(G, label, feat_blended_target, feature_idx, trunc_psi, noise_mode)
            
            # Re-run forward pass with optimized W+ (without blending hooks)
            img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
            
            # Save three images: before blending, after blending, after re-projection
            if self.save_blend_images and img_before_blend is not None and img_after_blend is not None:
                self._save_blending_three_images(img_before_blend, img_after_blend, img[0], self.drag_step)

        h, w = G.img_resolution, G.img_resolution

        if is_drag:
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            xx, yy = torch.meshgrid(X, Y)
            feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat[feature_idx].detach(), [h, w], mode='bilinear')
                self.feat_refs = []
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    self.feat_refs.append(self.feat0_resize[:,:,py,px])
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2

            # Point tracking with feature matching
            with torch.no_grad():
                for j, point in enumerate(points):
                    r = round(r2 / 512 * h)
                    up = max(point[0] - r, 0)
                    down = min(point[0] + r + 1, h)
                    left = max(point[1] - r, 0)
                    right = min(point[1] + r + 1, w)
                    feat_patch = feat_resize[:,:,up:down,left:right]
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                    _, idx = torch.min(L2.view(1,-1), -1)
                    width = right - left
                    point = [idx.item() // width + up, idx.item() % width + left]
                    points[j] = point

            res.points = [[point[0], point[1]] for point in points]

            # Motion supervision
            loss_motion = 0
            res.stop = True
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

            loss = loss_motion
            if mask is not None:
                if mask.min() == 0 and mask.max() == 1:
                    mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
                    loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
                    loss += lambda_mask * loss_fix

            loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
            if not res.stop:
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()
                
                # Increment drag step counter
                self.drag_step += 1
                
                # Save visualization after blending and re-projection
                # if should_blend:
                #     if self.save_blend_images and img_before_blend is not None:
                #         self._save_blend_visualization(img_before_blend, img[0], self.drag_step)

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img
        res.w = ws.detach().cpu().numpy()

#----------------------------------------------------------------------------
