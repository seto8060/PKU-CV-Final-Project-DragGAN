import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from pathlib import Path
import pickle



class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        print("ENTER SingleIDCoach.train")

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            if w_pivot.ndim == 3 and w_pivot.shape[1] != self.G.num_ws:
                print(f'[WARN] Align w_pivot from {w_pivot.shape[1]} to {self.G.num_ws}')
                w_pivot = w_pivot[:, :self.G.num_ws, :]

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(hyperparameters.max_pti_steps)):

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            w_pivot = w_pivot.to(global_config.device)

            pivot_path = (
                Path(paths_config.checkpoints_dir)
                / f'w_pivot_{global_config.run_name}_{paths_config.input_data_id}.pt'
            )
            torch.save(w_pivot.detach().cpu(), pivot_path)
            print(f'[OK] Saved w_pivot → {pivot_path}')
            with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
                d = pickle.load(f)
            
            old_D = d.get('D', None)
            
            snapshot = {
                'G': self.G.eval().requires_grad_(False).cpu(),
                'G_ema': self.G.eval().requires_grad_(False).cpu(),
                'D': old_D.eval().requires_grad_(False).cpu() if old_D is not None else None,
                'training_set_kwargs': None,
                'augment_pipe': None,
            }
            
            pkl_path = (
                Path(paths_config.checkpoints_dir)
                / f'model_{global_config.run_name}_{paths_config.input_data_id}.pkl'
            )
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(snapshot, f)
            
            print(f'[OK] Saved PTI-refined generator → {pkl_path}')

