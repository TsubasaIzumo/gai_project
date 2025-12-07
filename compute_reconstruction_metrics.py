#!/usr/bin/env python
# coding: utf-8
"""
只计算图像重建指标的简化版本
计算 L2, L1, PSNR, SSIM 指标，跳过所有源检测和catalog分析
"""

import argparse
import os
import re
import warnings

import tqdm
import torch
import numpy as np
import pandas as pd

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.utils import get_config
from src.data.dataset import SkaDataset, MakeDataLoader

warnings.simplefilter('ignore')


def compute_psnr_ssim(image1, image2):
    """计算 PSNR 和 SSIM"""
    # Convert images to grayscale if necessary
    if image1.ndim == 3:
        image1 = np.mean(image1, axis=2)
    if image2.ndim == 3:
        image2 = np.mean(image2, axis=2)

    # Compute PSNR
    dtype = image1.dtype
    image2 = image2.astype(dtype)

    # 明确指定data_range=1.0，因为图像已经被归一化到[0,1]
    psnr = peak_signal_noise_ratio(image1, image2, data_range=1.0)

    # Compute SSIM
    ssim = structural_similarity(image1, image2, data_range=1.0)

    # Return PSNR and SSIM values
    return psnr, ssim


def true_trasnform(true, power):
    """转换到显示空间"""
    const = (0.7063881) ** (30.0 / power)
    true = (true) ** (1. / power)
    true = (true) / const
    true = (true - 0.5) / 0.5
    return true


def true_itrasnform(true, power):
    """逆变换回物理空间"""
    const = (0.7063881) ** (30.0 / power)
    true_back = true * const
    true_back = (true_back) ** (power)
    return true_back


def im_reshape(downsampled_array):
    """将图像调整到512x512"""
    if downsampled_array.shape[1] == 512:
        return downsampled_array
    im_size = downsampled_array.shape[1]
    scale_factor = 512 // im_size
    downsampled_image = torch.tensor(downsampled_array).reshape(1, 1, im_size, im_size, )
    upsampled_image = torch.nn.functional.interpolate(downsampled_image, scale_factor=scale_factor, mode='bicubic')
    upsampled_image = upsampled_image.data.numpy()[0, 0, :, :, ]
    return upsampled_image


def add_column_i(sources, i):
    """添加索引列"""
    if sources is None:
        return None
    N = sources.shape[0]
    column_to_add = i * np.ones((N, 1))
    result = np.hstack((sources, column_to_add))
    return result


def compute_reconstruction_metrics_from_im(gen_im, im, verbose=False, power=None):
    """计算重建指标：L2, L1, PSNR, SSIM"""
    l2_dif = np.sqrt(np.sum(np.square(gen_im - im)))
    l1_dif = np.sum(np.abs(gen_im - im))
    image1 = true_trasnform(gen_im, power) / 2 + 0.5
    image2 = true_trasnform(im, power) / 2 + 0.5
    image1[image1 > 1] = 1
    image2[image2 > 1] = 1
    image1[image1 < 0] = 0
    image2[image2 < 0] = 0

    psnr, ssim = compute_psnr_ssim(image1, image2)

    if verbose:
        print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    reconstruction_metrics = [
        l2_dif,
        l1_dif,
        psnr,
        ssim,
    ]
    return reconstruction_metrics


def aggregate_images(images, aggregation='mean'):
    """聚合多个生成的图像"""
    if aggregation == 'mean':
        return np.mean(images, axis=0)
    if aggregation == 'median':
        return np.median(images, axis=0)
    if aggregation == 'medoid':
        if len(images) == 0:
            return None
        n = len(images)
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(images[i].flatten() - images[j].flatten())
        # Sum the distances
        sum_distances = np.sum(distances, axis=1)
        # Find the index of the medoid image
        medoid_idx = np.argmin(sum_distances)
        # Return the medoid image
        return images[medoid_idx]
    else:
        print("Such aggregation method is not implemented:", aggregation)
        return None


class ReconstructionMetricsCalculator:
    """只计算重建指标的类，跳过所有源检测"""
    
    def __init__(
            self,
            folder,
            dataset_folder,
            runs_per_sample,
            image_size=512,
            partition="test",
            test_size=0.5,
            random_state=2,
            real_data=False
    ):
        self.folder = folder
        self.dataset_folder = dataset_folder
        self.runs_per_sample = runs_per_sample
        self.image_size = image_size
        self.partition = partition
        self.power = self._extract_power_from_folder_name()
        self.test_size = test_size
        self.random_state = random_state
        self.real_data = real_data

        self.generated_images = []
        self.sky_indexes = []
        self.noisy_input = []
        self.sky_keys = []
        self.test_idx = None  # test集的索引列表（原始数据集中的索引）
        self.test_file_list = None  # test集的文件名列表
        self.load_data()

    def _extract_power_from_folder_name(self):
        """从文件夹名称中提取power值"""
        match = re.search(r'power(\d+(\.\d+)?)', self.folder)
        return float(match.group(1)) if match else 30

    def _extract_sample_index_from_filename(self, filename):
        """从文件名中提取样本索引
        文件名格式: batch=0000_sample=0000_run=00_latent=00.npy
        返回: sample_idx (整数)
        """
        try:
            # 提取 sample= 后面的数字
            match = re.search(r'sample=(\d+)', filename)
            if match:
                return int(match.group(1))
        except:
            pass
        return None

    def _is_new_format(self):
        """检查是否使用新格式（单个文件）"""
        all_files = [f for f in os.listdir(self.folder) if f.startswith("batch=") and f.endswith('.npy')]
        if not all_files:
            return False
        # 如果文件名包含 sample=，则是新格式
        return any('sample=' in f for f in all_files)

    def _compute_batch_nbs(self):
        """计算批次编号"""
        file_prefix = "batch="
        file_suffix = f"_{self.partition}_dirty_noisy.npy"
        all_files = [f for f in os.listdir(self.folder) if f.startswith(file_prefix) and f.endswith(file_suffix)]
        batch_numbers = [int(f[len(file_prefix):-len(file_suffix)]) for f in all_files]
        batch_numbers.sort()
        return batch_numbers

    def reorder_repeated(self, input_array):
        """重新排序重复的图像"""
        im_shape = input_array.shape[1:]
        N = input_array.shape[0]

        num_repeats = N // self.runs_per_sample

        reshaped_array = np.reshape(input_array, (self.runs_per_sample, num_repeats, *im_shape))
        transpose_axes = [1, 0] + list(range(2, len(im_shape) + 2))
        reshaped_array = np.transpose(reshaped_array, transpose_axes)
        reordered_array = np.reshape(reshaped_array, (N, *im_shape))

        return reordered_array
    
    def _load_and_preprocess_true_image(self, sky_index):
        """加载并预处理真实图像
        sky_index: test数据集中的索引（sample_idx）
        
        问题：sky_index是test数据集中的索引，但noisy_im_filenames是整个数据集的文件列表
        解决方案：使用test_idx来映射test集的索引到原始数据集的索引，然后获取文件名
        """
        try:
            filename = None
            
            # 方法1: 如果test_file_list可用，直接使用（最可靠的方法）
            if self.test_file_list is not None and sky_index < len(self.test_file_list):
                filename = self.test_file_list[sky_index]
            # 方法2: 如果test_idx可用，使用它来映射
            elif self.test_idx is not None and sky_index < len(self.test_idx):
                # test集中的第sky_index个样本对应原始数据集中的第test_idx[sky_index]个样本
                original_idx = self.test_idx[sky_index]
                if original_idx < len(self.noisy_im_filenames):
                    filename = self.noisy_im_filenames[original_idx]
            # 方法3: 使用sky_keys来映射（向后兼容）
            # 假设sky_keys的索引和test集的索引一致（对于real_data情况）
            elif sky_index < len(self.sky_keys):
                sky_key = self.sky_keys[sky_index]
                # sky_key可能是文件名或key
                # 如果sky_key就是文件名，直接使用
                if sky_key in self.noisy_im_filenames:
                    filename = sky_key
                else:
                    # 否则，假设sky_keys的索引和noisy_im_filenames的索引一致
                    # 这适用于real_data情况（test集包含所有样本）
                    if sky_index < len(self.noisy_im_filenames):
                        filename = self.noisy_im_filenames[sky_index]
                    else:
                        # 如果索引超出范围，尝试使用sky_key作为文件名的一部分来查找
                        matching_files = [f for f in self.noisy_im_filenames if str(sky_key) in f]
                        if matching_files:
                            filename = matching_files[0]
            # 方法4: 直接使用索引（向后兼容，适用于数据集未被分割的情况）
            else:
                if sky_index < len(self.noisy_im_filenames):
                    filename = self.noisy_im_filenames[sky_index]
            
            # 加载文件
            if filename is None:
                true = np.zeros((1, 512, 512))
            else:
                true_file = os.path.join(self.true_folder, filename)
                if os.path.exists(true_file):
                    true = np.load(true_file)[np.newaxis, ...]
                else:
                    # 如果文件不存在，返回零图像
                    true = np.zeros((1, 512, 512))
        except Exception as e:
            # 如果所有方法都失败，返回零图像
            true = np.zeros((1, 512, 512))
        
        # Apply the same preprocessing as in SkaDataset.__getitem__
        const = 0.00002960064
        true = true / const
        true = np.abs(true)
        true = (true) ** (1. / self.power)
        true = np.nan_to_num(true, nan=0.0, posinf=10.0, neginf=-10.0)
        true = np.clip(true, -10, 10)
        true = (true - 0.5) / 0.5
        
        # Convert to torch tensor and interpolate to target size
        sky_model = torch.from_numpy(true).float()
        sky_model = torch.clamp(sky_model, -10, 10)
        
        # Interpolate to target image size
        preprocessed = torch.nn.functional.interpolate(
            sky_model.unsqueeze(0),
            mode="bicubic",
            size=(self.image_size, self.image_size)
        )[0]
        
        # Convert back to numpy and return as (H, W) array
        preprocessed = preprocessed.squeeze(0).numpy()
        return preprocessed
    
    def load_data(self):
        """加载生成图像和真实图像数据 - 支持新格式和旧格式"""
        if self._is_new_format():
            # 新格式：从单个文件中加载
            print("检测到新格式（单个文件），正在加载...")
            self._load_data_new_format()
        else:
            # 旧格式：从批次文件中加载
            print("使用旧格式（批次文件），正在加载...")
            self._load_data_old_format()
        
        # 加载数据集信息
        self.sky_keys = np.load(f"{self.dataset_folder}/sky_keys.npy")
        self.true_folder = f"{self.dataset_folder}/true"
        self.noisy_folder = f"{self.dataset_folder}/dirty"
        self.noisy_im_filenames = os.listdir(self.noisy_folder)
        self.noisy_im_filenames.sort()
        
        # 创建test数据集以获取test集的索引映射
        # 这样可以将test集的索引映射到原始数据集的文件名
        try:
            from sklearn.model_selection import train_test_split
            
            # 创建原始数据集
            dataset = SkaDataset(self.dataset_folder, self.image_size, self.power, from_uv=False)
            original_size = len(dataset)
            
            # 计算test集的索引（与训练时使用相同的分割参数）
            if self.real_data:
                # 对于real_data，test集包含所有样本
                self.test_idx = list(range(original_size))
            else:
                # 使用与训练时相同的分割参数
                train_idx, test_idx = train_test_split(
                    list(range(original_size)), 
                    test_size=self.test_size,
                    random_state=self.random_state
                )
                valid_idx, test_idx = train_test_split(
                    test_idx, 
                    test_size=0.5, 
                    random_state=self.random_state + 1
                )
                self.test_idx = test_idx
            
            # 创建test集文件名列表的映射
            # test集中的第i个样本对应原始数据集中的第test_idx[i]个样本
            self.test_file_list = [dataset.file_list[idx] for idx in self.test_idx]
            
        except Exception as e:
            print(f"警告: 无法创建test数据集来获取文件名映射: {e}")
            print(f"  将使用sky_keys来映射（假设test集的索引和原始数据集的索引一致）")
            self.test_idx = None
            self.test_file_list = None

    def _load_data_new_format(self):
        """加载新格式的数据（单个文件）"""
        all_files = [f for f in os.listdir(self.folder) 
                    if f.startswith("batch=") and f.endswith('.npy') and 'sample=' in f]
        
        # 按 sample_idx 和 run_idx 排序
        file_info = []
        for f in all_files:
            sample_idx = self._extract_sample_index_from_filename(f)
            if sample_idx is not None:
                # 提取 run_idx 和 latent_idx
                run_match = re.search(r'run=(\d+)', f)
                latent_match = re.search(r'latent=(\d+)', f)
                run_idx = int(run_match.group(1)) if run_match else 0
                latent_idx = int(latent_match.group(1)) if latent_match else 0
                file_info.append({
                    'filename': f,
                    'sample_idx': sample_idx,
                    'run_idx': run_idx,
                    'latent_idx': latent_idx,
                    'full_path': os.path.join(self.folder, f)
                })
        
        # 按 sample_idx, run_idx, latent_idx 排序
        file_info.sort(key=lambda x: (x['sample_idx'], x['run_idx'], x['latent_idx']))
        
        # 不在这里限制样本数量，在compute_all_metrics中根据max_samples参数限制
        unique_samples = sorted(set([f['sample_idx'] for f in file_info]))
        
        print(f"找到 {len(unique_samples)} 个唯一样本")
        print(f"总共 {len(file_info)} 个文件")
        
        # 加载图像数据
        generated_images_list = []
        sky_indexes_list = []
        
        for file_item in file_info:
            img_data = np.load(file_item['full_path'])
            # 确保是2D或3D数组
            if img_data.ndim == 2:
                img_data = img_data[np.newaxis, ...]  # 添加通道维度
            elif img_data.ndim == 3:
                # 如果是 (C, H, W)，保持不变
                pass
            else:
                print(f"警告: 文件 {file_item['filename']} 的形状 {img_data.shape} 不符合预期")
                continue
            
            generated_images_list.append(img_data)
            # 保存文件名用于后续提取索引
            sky_indexes_list.append(file_item['filename'])
        
        if generated_images_list:
            self.generated_images = np.array(generated_images_list)
            self.sky_indexes = np.array(sky_indexes_list, dtype=object)
            # 对于新格式，noisy_input 可能不存在，设为空列表
            self.noisy_input = []
        else:
            raise ValueError("没有找到有效的图像文件")

    def _load_data_old_format(self):
        """加载旧格式的数据（批次文件）"""
        if self.runs_per_sample != -1:
            batch_numbers = self._compute_batch_nbs()
            for i in batch_numbers:
                line = f"batch={i}_"
                test_generated_images_i = np.load(f"{self.folder}/{line}{self.partition}_generated_images.npy")
                sky_indexes_i = np.load(f"{self.folder}/{line}{self.partition}_sky_indexes.npy")
                noisy_input_i = np.load(f"{self.folder}/{line}{self.partition}_dirty_noisy.npy")

                test_generated_images_i = self.reorder_repeated(test_generated_images_i)
                sky_indexes_i = self.reorder_repeated(sky_indexes_i)
                noisy_input_i = self.reorder_repeated(noisy_input_i)

                self.generated_images.append(test_generated_images_i)
                self.sky_indexes.append(sky_indexes_i)
                self.noisy_input.append(noisy_input_i)

            self.generated_images = np.concatenate(self.generated_images)
            self.sky_indexes = np.concatenate(self.sky_indexes)
            self.noisy_input = np.concatenate(self.noisy_input)

    def compute_reconstruction_metrics_for_sample(
            self,
            i,
            verbose=False,
            apply_itransform=True,
            aggr="median",
    ):
        """计算单个样本的重建指标"""
        # 从文件名中提取 sky_index
        if isinstance(self.sky_indexes[i], str):
            # 新格式：从文件名中提取
            sky_index = self._extract_sample_index_from_filename(self.sky_indexes[i])
            if sky_index is None:
                print(f"警告: 无法从文件名 {self.sky_indexes[i]} 中提取样本索引")
                sky_index = 0
        else:
            # 旧格式：从字符串中提取
            sky_index = int(self.sky_indexes[i][-9:-4])
        
        try:
            # Load and preprocess true image
            im = self._load_and_preprocess_true_image(sky_index)
        except Exception as e:
            if verbose:
                print(f"加载真实图像失败 (sky_index={sky_index}): {e}")
            im = np.zeros((self.image_size, self.image_size))

        # 对于新格式，需要根据实际的文件组织方式来确定重复次数
        if self._is_new_format():
            # 找到所有属于同一个 sample_idx 的文件
            same_sample_files = [idx for idx in range(len(self.sky_indexes)) 
                                if isinstance(self.sky_indexes[idx], str) and 
                                self._extract_sample_index_from_filename(self.sky_indexes[idx]) == sky_index]
            # 如果设置了runs_per_sample，限制文件数量
            if self.runs_per_sample > 0:
                same_sample_files = same_sample_files[:self.runs_per_sample]
            repeat_images = len(same_sample_files) if same_sample_files else 1
        else:
            # 旧格式：按顺序处理
            repeat_images = self.runs_per_sample
            same_sample_files = [i + j for j in range(repeat_images) if i + j < len(self.generated_images)]
        
        reconstruction_metrics_generated_all_runs = []
        gen_ims = []

        # 对每个生成的图像计算重建指标
        for j, idx in enumerate(same_sample_files):
            if idx >= len(self.generated_images):
                break
                
            gen_im = self.generated_images[idx]
            # 处理不同的数组维度
            if gen_im.ndim == 3:
                gen_im = gen_im[0, :, :]  # 取第一个通道
            elif gen_im.ndim == 2:
                pass  # 已经是2D
            else:
                if verbose:
                    print(f"警告: 图像维度 {gen_im.ndim} 不符合预期")
                continue
                
            gen_im = im_reshape(gen_im)

            if apply_itransform:
                gen_im_astro = true_itrasnform(gen_im, self.power)

            # 计算重建指标
            im_astro = true_itrasnform(im, self.power)
            reconstruction_metrics_generated = compute_reconstruction_metrics_from_im(
                gen_im_astro, im_astro, power=self.power, verbose=verbose
            )
            # 使用j作为diff_idx
            reconstruction_metrics_generated = reconstruction_metrics_generated + [j, ]
            reconstruction_metrics_generated_all_runs.append(reconstruction_metrics_generated)
            gen_ims.append(gen_im)

        if len(gen_ims) == 0:
            gen_ims = [gen_im, ]

        # 聚合多个生成的图像
        ims_array = np.array(gen_ims)
        gen_im = aggregate_images(ims_array, aggregation=aggr)

        if apply_itransform:
            gen_im_astro = true_itrasnform(gen_im, self.power)

        # 计算聚合后的重建指标
        im_astro = true_itrasnform(im, self.power)
        reconstruction_metrics_generated = compute_reconstruction_metrics_from_im(
            gen_im_astro, im_astro, verbose=False, power=self.power
        )
        reconstruction_metrics_generated = reconstruction_metrics_generated + [-1, ]
        reconstruction_metrics_generated_all_runs.append(reconstruction_metrics_generated)
        reconstruction_metrics_generated_all_runs = np.array(reconstruction_metrics_generated_all_runs)

        data = {}
        # 对于新格式，需要正确计算样本索引
        if self._is_new_format():
            sample_idx_for_column = sky_index
        else:
            sample_idx_for_column = i // self.runs_per_sample
            
        data["reconstruction_metrics"] = add_column_i(
            reconstruction_metrics_generated_all_runs, 
            sample_idx_for_column
        )
        return data

    def compute_all_metrics(
            self,
            verbose=False,
            apply_itransform=True,
            aggr="median",
            max_samples=45,
    ):
        """计算所有样本的重建指标"""
        reconstruction_metrics = []

        if self._is_new_format():
            # 新格式：需要按样本分组处理
            # 获取所有唯一的样本索引
            unique_samples = []
            for i in range(len(self.sky_indexes)):
                if isinstance(self.sky_indexes[i], str):
                    sample_idx = self._extract_sample_index_from_filename(self.sky_indexes[i])
                    if sample_idx is not None and sample_idx not in unique_samples:
                        unique_samples.append(sample_idx)
            
            unique_samples.sort()
            # 限制为前max_samples个
            unique_samples = unique_samples[:max_samples]
            
            print(f"处理 {len(unique_samples)} 个样本（限制为前 {max_samples} 个）")
            
            # 对每个样本，找到所有相关的文件索引
            for sample_idx in tqdm.tqdm(unique_samples, desc="计算指标"):
                # 找到所有属于这个样本的文件索引
                file_indices = [i for i in range(len(self.sky_indexes)) 
                              if self._extract_sample_index_from_filename(self.sky_indexes[i]) == sample_idx]
                
                if not file_indices:
                    continue
                
                # 使用第一个文件索引作为代表
                i = file_indices[0]
                data = self.compute_reconstruction_metrics_for_sample(
                    i,
                    verbose,
                    apply_itransform,
                    aggr
                )
                reconstruction_metrics.append(data["reconstruction_metrics"])
        else:
            # 旧格式：按原来的方式处理
            # 计算总样本数（考虑runs_per_sample）
            total_samples = len(self.generated_images) // self.runs_per_sample
            # 限制为最多max_samples个样本
            num_samples_to_process = min(max_samples, total_samples)
            
            print(f"总共找到 {total_samples} 个样本，将处理前 {num_samples_to_process} 个样本")
            
            for i in tqdm.tqdm(range(0, num_samples_to_process * self.runs_per_sample, self.runs_per_sample)):
                data = self.compute_reconstruction_metrics_for_sample(
                    i,
                    verbose,
                    apply_itransform,
                    aggr
                )
                reconstruction_metrics.append(data["reconstruction_metrics"])

        if len(reconstruction_metrics) > 0:
            reconstruction_metrics = np.vstack(reconstruction_metrics)
        else:
            reconstruction_metrics = np.array([])

        return {"reconstruction_metrics": reconstruction_metrics}


def main(folders, dataset_folder, runs_per_sample, image_size=512, partition="test", max_samples=45,
         test_size=0.5, random_state=2, real_data=False):
    """主函数：只计算和保存重建指标"""
    for folder in folders:
        print(f"\n处理文件夹: {folder}")
        calculator = ReconstructionMetricsCalculator(
            folder, dataset_folder, runs_per_sample, image_size, partition,
            test_size=test_size, random_state=random_state, real_data=real_data
        )
        reconstruction_columns = ["l2", "l1", "psnr", "ssim"]
        
        for aggr in ["mean", "medoid", "median"]:
            current_key = aggr
            print(f"\n计算 {current_key} 聚合方式的重建指标...")
            res = calculator.compute_all_metrics(aggr=current_key, max_samples=max_samples)

            if len(res["reconstruction_metrics"]) == 0:
                print(f"警告: 没有找到重建指标数据")
                continue

            # 保存重建指标
            reconstruction_metrics = pd.DataFrame(
                res["reconstruction_metrics"], 
                columns=reconstruction_columns + ["diff_idx",] + ["image_idx",]
            )
            reconstruction_metrics.to_csv(folder + f"/{current_key}_reconstruction_metrics.csv", index=False)
            
            # 打印统计信息
            aggregated_metrics = reconstruction_metrics[reconstruction_metrics['diff_idx'] == -1]
            if len(aggregated_metrics) > 0:
                print(f"  已保存重建指标到: {folder}/{current_key}_reconstruction_metrics.csv")
                print(f"  处理的样本数: {len(aggregated_metrics)}")
                print(f"  平均 L2: {aggregated_metrics['l2'].mean():.6f}")
                print(f"  平均 L1: {aggregated_metrics['l1'].mean():.6f}")
                print(f"  平均 PSNR: {aggregated_metrics['psnr'].mean():.2f}")
                print(f"  平均 SSIM: {aggregated_metrics['ssim'].mean():.4f}")
            else:
                print(f"  警告: 没有找到聚合后的指标 (diff_idx=-1)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只计算图像重建指标 (L2, L1, PSNR, SSIM)')
    parser.add_argument('--config', '-c', type=str,
                        default='./configs/generate_lightweight.yaml',
                        help='Path to config')
    parser.add_argument('--runs_per_sample',
                        type=int, default=-1,
                        help='每个样本的运行次数')
    parser.add_argument('--folders', nargs='+', required=True, default='E:\astroDDPM\generated_pretrained_20251205_190211_power2',
                        help='要处理的文件夹列表')
    parser.add_argument('--max_samples', type=int, default=235,
                        help='最大处理的样本数量（默认：45）')
    parser.add_argument('--test_size', type=float, default=0.5,
                        help='测试集大小比例（默认：0.5）')
    parser.add_argument('--random_state', type=int, default=2,
                        help='随机种子（默认：2）')
    parser.add_argument('--real_data', action='store_true',
                        help='是否使用真实数据（默认：False）')
    args = parser.parse_args()
    config = get_config(args.config)

    folders = args.folders
    main(
        folders, 
        dataset_folder=config["dataset"]["image_path"], 
        runs_per_sample=args.runs_per_sample,
        max_samples=args.max_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        real_data=args.real_data
    )

