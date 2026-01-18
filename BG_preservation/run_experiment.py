import os
import os.path as osp
import json
import argparse
from PIL import Image
import numpy as np
import torch

import dnnlib
from viz.renderer import Renderer, add_watermark_np
from gradio_utils import draw_points_on_image, draw_mask_on_image


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def create_annotated_image(image, points, mask, show_mask=True):
    """Create an image with points and mask annotations, similar to visualizer_drag_gradio.py"""
    # Start with the original image
    annotated_image = draw_points_on_image(image, points)

    # Add mask overlay if mask is provided and show_mask is True
    if show_mask and mask is not None and not (mask == 0).all() and not (mask == 1).all():
        annotated_image = draw_mask_on_image(annotated_image, mask)

    return annotated_image


def load_experiment_data(filepath):
    """Load experiment configuration from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def setup_global_state(experiment_data, cache_dir):
    """Setup global state from experiment data"""
    valid_checkpoints_dict = {
        f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
    }

    global_state = {
        "images": {},
        "temporal_params": {
            "stop": False,
        },
        'mask': np.array(experiment_data['mask']) if experiment_data['mask'] else None,
        'last_mask': None,
        'show_mask': experiment_data['show_mask'],
        "generator_params": dnnlib.EasyDict(),
        "params": experiment_data['params'],
        "device": 'cuda',
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": experiment_data['points'],
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': experiment_data.get('editing_state', 'add_points'),
        'pretrained_weight': experiment_data['pretrained_weight']
    }

    return global_state, valid_checkpoints_dict


def init_network(global_state, valid_checkpoints_dict, blend_interval=50, reproject_steps=50):
    """Initialize the network with experiment parameters"""
    state = global_state

    state['renderer'].init_network(
        state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['seed'],  # w0_seed,
        None,  # w_load
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    # Set re-projection parameters
    state['renderer'].blend_interval = blend_interval
    state['renderer'].reproject_steps = reproject_steps

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image

    return global_state


def run_drag_experiment(global_state, num_steps=201, save_interval=50):
    """Run the drag experiment for specified number of steps"""

    p_in_pixels = []
    t_in_pixels = []
    valid_points = []

    # Prepare the points for the inference
    for key_point, point in global_state["points"].items():
        try:
            p_start = point.get("start_temp", point["start"])
            p_end = point["target"]

            if p_start is None or p_end is None:
                continue

        except KeyError:
            continue

        p_in_pixels.append(p_start)
        t_in_pixels.append(p_end)
        valid_points.append(key_point)

    if len(p_in_pixels) == 0:
        print("No valid points found for dragging!")
        return global_state, []

    mask = torch.tensor(global_state['mask']).float()
    drag_mask = 1 - mask

    renderer = global_state["renderer"]

    # reverse points order
    p_to_opt = reverse_point_pairs(p_in_pixels)
    t_to_opt = reverse_point_pairs(t_in_pixels)

    print('Running experiment with:')
    print(f'    Source: {p_in_pixels}')
    print(f'    Target: {t_in_pixels}')
    print(f'    Total steps: {num_steps}')

    results = []
    step_idx = 0

    # Store initial image for metrics calculation
    initial_image = global_state['images']['image_raw'].copy()

    while step_idx < num_steps:
        # do drag here!
        renderer._render_drag_impl(
            global_state['generator_params'],
            p_to_opt,  # point
            t_to_opt,  # target
            drag_mask,  # mask,
            global_state['params']['motion_lambda'],  # lambda_mask
            reg=0,
            feature_idx=5,  # NOTE: do not support change for now
            r1=global_state['params']['r1_in_pixels'],  # r1
            r2=global_state['params']['r2_in_pixels'],  # r2
            trunc_psi=global_state['params']['trunc_psi'],
            is_drag=True,
            to_pil=True)

        # Update points position
        for key_point, p_i, t_i in zip(valid_points, p_to_opt, t_to_opt):
            global_state["points"][key_point]["start_temp"] = [
                p_i[1],
                p_i[0],
            ]
            global_state["points"][key_point]["target"] = [
                t_i[1],
                t_i[0],
            ]

        image_result = global_state['generator_params']['image']
        global_state['images']['image_raw'] = image_result

        # Save intermediate results
        if step_idx % save_interval == 0 or step_idx == num_steps - 1:
            print(f'Step {step_idx}: Source positions updated')
            current_points = {}
            for key_point in valid_points:
                current_points[key_point] = {
                    'start': global_state["points"][key_point]["start_temp"],
                    'target': global_state["points"][key_point]["target"]
                }

            results.append({
                'step': step_idx,
                'image': image_result,
                'points': current_points
            })

        step_idx += 1

    print(f'Experiment completed after {num_steps} steps')
    return global_state, results, initial_image


def save_experiment_results(experiment_data, results, output_dir, initial_image, total_steps, blend_interval, reproject_steps):
    """Save experiment results to output directory"""

    os.makedirs(output_dir, exist_ok=True)

    # Save final images (both clean and annotated versions)
    final_result = results[-1] if results else None
    if final_result:
        final_image = final_result['image']

        # Save clean final image
        final_clean_path = os.path.join(output_dir, 'final_result_clean.png')
        final_image.save(final_clean_path)
        print(f'Final clean image saved to: {final_clean_path}')

        # Save annotated final image with points and mask
        final_annotated_image = create_annotated_image(
            final_image,
            final_result['points'],
            np.array(experiment_data['mask']) if experiment_data['mask'] else None,
            experiment_data['show_mask']
        )
        final_annotated_path = os.path.join(output_dir, 'final_result_annotated.png')
        final_annotated_image.save(final_annotated_path)
        print(f'Final annotated image saved to: {final_annotated_path}')

        # Calculate experiment metrics
        print('Calculating experiment metrics...')
        metrics = calculate_experiment_metrics(
            initial_image,
            final_image,
            np.array(experiment_data['mask']) if experiment_data['mask'] else None,
            final_result['points'],
            experiment_data['points']
        )

        # Save metrics and re-projection parameters
        points_data = {
            'metrics': metrics,
            'total_steps': total_steps,
            'blend_interval': blend_interval,
            'reproject_steps': reproject_steps
        }

        points_path = os.path.join(output_dir, 'experiment_report.json')
        with open(points_path, 'w', encoding='utf-8') as f:
            json.dump(points_data, f, indent=2, ensure_ascii=False)
        print(f'Experiment report saved to: {points_path}')

        # Save intermediate results (both clean and annotated versions)
        for i, result in enumerate(results[:-1]):  # Don't save the final one again
            step = result['step']

            # Save clean image
            clean_img_path = os.path.join(output_dir, f'step_{step:04d}_clean.png')
            result['image'].save(clean_img_path)

            # Save annotated image with points and mask
            annotated_image = create_annotated_image(
                result['image'],
                result['points'],
                np.array(experiment_data['mask']) if experiment_data['mask'] else None,
                experiment_data['show_mask']
            )
            annotated_img_path = os.path.join(output_dir, f'step_{step:04d}_annotated.png')
            annotated_image.save(annotated_img_path)

        print(f'Intermediate results saved to: {output_dir}')
        print(f'Metrics calculated:')
        print(f'  - External pixel change: {metrics.get("external_pixel_change", "N/A")}')
        print(f'  - Inside mask movement ratio: {metrics.get("inside_mask_movement_ratio", "N/A")}')
        print(f'  - Outside mask movement ratio: {metrics.get("outside_mask_movement_ratio", "N/A")}')


def calculate_experiment_metrics(initial_image, final_image, mask, points_data, original_points):
    """Calculate experiment metrics"""
    metrics = {}

    # Convert images to numpy arrays
    init_array = np.array(initial_image)
    final_array = np.array(final_image)

    if mask is not None:
        mask_array = np.array(mask)
    else:
        # If no mask, assume all pixels are outside
        mask_array = np.zeros((init_array.shape[0], init_array.shape[1]))

    # 1. Mask外部像素点的变化情况 (External pixel changes)
    # mask为0表示外部，1表示内部
    external_mask = 1 - mask_array  # 外部像素为1，内部像素为0

    # 计算外部像素的变化 (color difference squared)
    pixel_diff = (init_array.astype(float) - final_array.astype(float)) ** 2
    # 只计算RGB通道的变化，忽略alpha通道如果存在
    if pixel_diff.shape[-1] > 3:
        pixel_diff = pixel_diff[:, :, :3]

    # 使用外部mask进行点乘
    external_changes = pixel_diff * external_mask[:, :, np.newaxis]
    total_external_change = np.sum(external_changes)

    # 计算外部像素的数量
    external_pixel_count = np.sum(external_mask)

    # 计算平均外部像素变化
    if external_pixel_count > 0:
        avg_external_change = total_external_change / external_pixel_count
        metrics['external_pixel_change'] = float(avg_external_change)
    else:
        metrics['external_pixel_change'] = 0.0

    # 2. 移动程度计算 (Movement ratios)
    points_inside_mask = []
    points_outside_mask = []

    for point_key, point_info in original_points.items():
        start_pos = point_info["start"]
        target_pos = point_info["target"]

        # 检查点的初始位置是否在mask内 (mask值为1表示内部)
        if mask is not None:
            # 确保坐标在图像范围内
            x, y = int(start_pos[0]), int(start_pos[1])
            if 0 <= x < mask_array.shape[1] and 0 <= y < mask_array.shape[0]:
                is_inside = mask_array[y, x] > 0.5  # 使用0.5作为阈值，因为mask可能被平滑
            else:
                is_inside = False
        else:
            is_inside = False

        # 计算移动程度
        if point_key in points_data:
            current_start = points_data[point_key]["start"]
            target = points_data[point_key]["target"]

            # 计算初始距离
            initial_dist = np.sqrt((start_pos[0] - target_pos[0])**2 + (start_pos[1] - target_pos[1])**2)
            # 计算最终距离
            final_dist = np.sqrt((current_start[0] - target[0])**2 + (current_start[1] - target[1])**2)

            if initial_dist > 0:
                movement_ratio = 1 - (final_dist / initial_dist)
            else:
                movement_ratio = 1.0  # 如果初始距离为0，说明已经在目标位置，移动程度为1

            point_data = {
                'point_key': point_key,
                'initial_distance': float(initial_dist),
                'final_distance': float(final_dist),
                'movement_ratio': float(movement_ratio)
            }

            if is_inside:
                points_inside_mask.append(point_data)
            else:
                points_outside_mask.append(point_data)

    # 计算平均移动程度
    if points_inside_mask:
        avg_inside_ratio = np.mean([p['movement_ratio'] for p in points_inside_mask])
        metrics['inside_mask_movement_ratio'] = float(avg_inside_ratio)
        metrics['inside_mask_points'] = points_inside_mask
    else:
        metrics['inside_mask_movement_ratio'] = None
        metrics['inside_mask_points'] = []

    if points_outside_mask:
        avg_outside_ratio = np.mean([p['movement_ratio'] for p in points_outside_mask])
        metrics['outside_mask_movement_ratio'] = float(avg_outside_ratio)
        metrics['outside_mask_points'] = points_outside_mask
    else:
        metrics['outside_mask_movement_ratio'] = None
        metrics['outside_mask_points'] = []

    return metrics


def run_single_experiment(config_file, output_dir, steps, save_interval, blend_interval, reproject_steps, cache_dir):
    """Run a single experiment"""
    print(f'\n=== Running experiment: {os.path.basename(config_file)} ===')

    # Load experiment configuration
    print(f'Loading experiment configuration from: {config_file}')
    experiment_data = load_experiment_data(config_file)

    print(f'Results will be saved to: {output_dir}')
    print(f'Using blend_interval={blend_interval}, reproject_steps={reproject_steps}')

    # Setup global state
    print('Setting up global state...')
    global_state, valid_checkpoints_dict = setup_global_state(experiment_data, cache_dir)

    # Initialize network
    print('Initializing network...')
    global_state = init_network(global_state, valid_checkpoints_dict, blend_interval, reproject_steps)

    # Run experiment
    print(f'Running experiment for {steps} steps...')
    global_state, results, initial_image = run_drag_experiment(global_state, steps, save_interval)

    # Save results
    print('Saving experiment results...')
    save_experiment_results(experiment_data, results, output_dir, initial_image, steps, blend_interval, reproject_steps)

    print(f'Experiment {os.path.basename(config_file)} completed successfully!')
    print(f'Results saved in: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Run DragGAN experiments from configuration files')
    parser.add_argument('output_base_dir', help='Base output directory for all experiment results')
    parser.add_argument('--experiment-data-dir', '-d', default='./experiment_data',
                        help='Directory containing experiment configuration JSON files (default: ./experiment_data)')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                        help='Specific experiment configuration file to run (if not specified, runs all experiments in --experiment-data-dir)')
    parser.add_argument('--steps', '-s', type=int, default=201,
                        help='Number of steps to run for each experiment (default: 201)')
    parser.add_argument('--save-interval', '-i', type=int, default=50,
                        help='Save intermediate results every N steps (default: 50). Saves both clean and annotated images.')
    parser.add_argument('--blend-interval', '-b', type=int, default=50,
                        help='Blend interval N: perform feature blending every N steps (default: 50)')
    parser.add_argument('--reproject-steps', '-r', type=int, default=25,
                        help='Re-projection steps M: number of re-projection steps to perform (default: 25)')
    parser.add_argument('--cache-dir', type=str, default='/root/cv-final-project/DragGAN/checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--pattern', '-p', default='*.json',
                        help='File pattern to match experiment files (default: *.json)')

    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Determine which experiments to run
    if args.experiment:
        # Run specific experiment
        if os.path.isabs(args.experiment):
            config_file = args.experiment
        else:
            config_file = os.path.join(args.experiment_data_dir, args.experiment)

        if not os.path.exists(config_file):
            print(f'Error: Specified experiment file does not exist: {config_file}')
            return

        experiment_files = [config_file]
        print(f'Running specific experiment: {os.path.basename(config_file)}')
    else:
        # Find all experiment configuration files
        experiment_files = []
        for file in os.listdir(args.experiment_data_dir):
            if file.endswith('.json'):
                experiment_files.append(os.path.join(args.experiment_data_dir, file))

        if not experiment_files:
            print(f'No experiment configuration files found in {args.experiment_data_dir}')
            return

        print(f'Found {len(experiment_files)} experiment configuration files:')
        for exp_file in experiment_files:
            print(f'  - {os.path.basename(exp_file)}')

    # Run each experiment
    for config_file in experiment_files:
        # Create experiment-specific output directory
        exp_name = os.path.splitext(os.path.basename(config_file))[0]
        exp_output_dir = os.path.join(args.output_base_dir, exp_name)

        try:
            run_single_experiment(
                config_file=config_file,
                output_dir=exp_output_dir,
                steps=args.steps,
                save_interval=args.save_interval,
                blend_interval=args.blend_interval,
                reproject_steps=args.reproject_steps,
                cache_dir=args.cache_dir
            )
        except Exception as e:
            print(f'Error running experiment {exp_name}: {str(e)}')
            import traceback
            traceback.print_exc()
            continue

    print(f'\n=== All experiments completed! ===')
    print(f'Results saved in: {args.output_base_dir}')


if __name__ == '__main__':
    main()