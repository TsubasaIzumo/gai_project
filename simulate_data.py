# CASA Simulation Script based on Table 1 parameters

import os
import numpy as np
import shutil
import uuid
from datetime import datetime
import pickle

# 修改为用户确定有权限的目录
base_dir = os.path.expanduser("~/data/casa_workspace/simulated_data")
print(f"Using base directory: {base_dir}")

# 创建数据和numpy子目录
data_dir = os.path.join(base_dir, "data")
numpy_dir = os.path.join(base_dir, "numpy_arrays")
true_dir = os.path.join(numpy_dir, "true")
dirty_dir = os.path.join(numpy_dir, "dirty")

# 确保目录存在
for directory in [base_dir, data_dir, numpy_dir, true_dir, dirty_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# 清理之前的运行结果，避免文件冲突
print("Cleaning previous simulation results...")
for directory in [data_dir, numpy_dir, true_dir, dirty_dir]:
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

# 存储所有模拟的sky_keys和ra_dec信息
all_sky_keys = []
all_ra_dec = {}  # 改为字典格式，与样本一致

# 存储所有射电源信息的字典
all_sources_info = {}

print("Starting ALMA simulation based on Table 1 parameters")

# Set parameters from Table 1
field_name = "COSMOS"
field_coords = "J2000 10h00m28.6s +02d12m21.0s"
num_pointings = 9164
num_antennas = 50
central_freq = "230GHz"  # Band 6
channel_width = 7.8  # MHz
num_channels = 240
total_bandwidth = channel_width * num_channels  # MHz
integration_time = "10s"
total_time = "20min"
hour_angle = "transit"

# For practical purposes, limit the number of simulations
max_sims = 10

# 添加时间戳，确保文件名唯一
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存当前工作目录
original_dir = os.getcwd()

# 计算COSMOS场内合理的坐标范围（约1度范围）
ra_base = 150.1192  # COSMOS 中心RA
dec_base = 2.2058  # COSMOS 中心DEC

# Run simulations
for sim_idx in range(max_sims):
    print(f"Running simulation {sim_idx + 1}/{max_sims}")

    # Project name - 只使用名称，不包含路径
    project_name = f"sim_{timestamp}_{sim_idx}"

    # 创建UUID格式的sky_key并保存
    sky_key = str(uuid.uuid4())
    all_sky_keys.append(sky_key)

    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 创建组件列表路径
    cl_file = os.path.join(data_dir, f"{project_name}.cl")

    # 确保组件列表不存在
    if os.path.exists(cl_file):
        os.system(f"rm -rf {cl_file}")

    cl.done()  # Clean up any existing list

    # 为每个模拟生成不同的相位中心，在COSMOS场内合理范围内
    # 生成随机偏移量，范围约±0.5度
    ra_offset = np.random.uniform(-0.6, 0.6)
    dec_offset = np.random.uniform(-0.6, 0.6)

    sim_ra = ra_base + ra_offset
    sim_dec = dec_base + dec_offset

    # 将相位中心转换为CASA格式的坐标字符串
    sim_ra_hms = f"{int(sim_ra / 15)}h{int((sim_ra / 15 % 1) * 60):02d}m{((sim_ra / 15 % 1) * 60 % 1) * 60:.1f}s"
    sim_dec_dms = f"{'+' if sim_dec >= 0 else '-'}{abs(int(sim_dec))}d{int(abs(sim_dec) % 1 * 60):02d}m{(abs(sim_dec) % 1 * 60 % 1) * 60:.1f}s"
    sim_coords = f"J2000 {sim_ra_hms} {sim_dec_dms}"

    # 保存相位中心（用字典格式）
    all_ra_dec[sky_key] = {'RA': sim_ra, 'DEC': sim_dec}

    print(f"Using phase center: RA={sim_ra}, DEC={sim_dec}")

    # 随机生成1-5个源，确保至少有1个源
    num_sources = np.random.randint(1, 6)
    sources_info = []

    for src_idx in range(num_sources):
        # Generate random source properties
        flux = np.random.uniform(0.05, 0.5) / 1000.0  # mJy to Jy
        major_axis = np.random.uniform(0.4, 0.8)  # arcsec
        minor_axis = np.random.uniform(0.4, major_axis)  # arcsec
        pa = np.random.uniform(0, 360)  # degrees

        # Random position within primary beam
        radius_arcsec = np.random.uniform(0, 22.86 / 2)
        angle_rad = np.random.uniform(0, 2 * np.pi)

        # Convert to RA/DEC offsets
        ra_src_offset = radius_arcsec * np.cos(angle_rad) / 3600.0  # Convert arcsec to degrees
        dec_src_offset = radius_arcsec * np.sin(angle_rad) / 3600.0

        # 计算SNR（这里是简化估计，实际需要考虑更多因素）
        # 假设噪声水平为1uJy/beam
        noise_level = 1e-6  # Jy/beam
        snr = flux / noise_level

        # 归一化SNR（考虑源的大小）
        beam_maj = 0.89  # arcsec - 典型ALMA beam size
        beam_min = 0.82  # arcsec
        snr_normalized = snr * (beam_maj * beam_min) / np.sqrt(
            (beam_maj ** 2 + major_axis ** 2) * (beam_min ** 2 + minor_axis ** 2))

        # 添加组件到模型 - 使用模拟特定的相位中心
        cl.addcomponent(
            dir=sim_coords,
            flux=flux,
            fluxunit="Jy",
            freq=central_freq,
            shape="gaussian",
            majoraxis=f"{major_axis}arcsec",
            minoraxis=f"{minor_axis}arcsec",
            positionangle=f"{pa}deg",
            spectrumtype="spectral index",
            index=0
        )

        # 计算源的实际RA和DEC
        ra = sim_ra + ra_src_offset
        dec = sim_dec + dec_src_offset

        sources_info.append([ra, dec, snr, snr_normalized, flux, major_axis, minor_axis])

    # 保存组件列表
    cl.rename(cl_file)
    cl.done()

    # 保存射电源信息
    all_sources_info[sky_key] = sources_info

    # 切换到数据目录
    os.chdir(data_dir)

    # 使用更合适的频谱设置进行模拟 - 使用模拟特定的相位中心
    simobserve(
        project=project_name,
        skymodel="",
        complist=cl_file,
        setpointings=True,
        integration=integration_time,
        direction=sim_coords,  # 使用模拟特定的相位中心
        mapsize="22.86arcsec",
        obsmode="int",
        totaltime=total_time,
        antennalist="alma.cycle5.3.cfg",
        hourangle=hour_angle,
        thermalnoise="tsys-atm",
        user_pwv=1.796,
        graphics="none",
        incenter=central_freq,
        inwidth=f"{total_bandwidth}MHz",
        incell="0.1arcsec",
        inbright="0.5mJy/beam",
        overwrite=True
    )

    # simobserve创建的项目目录和MS文件路径
    project_dir = os.path.join(data_dir, project_name)
    ms_file = os.path.join(project_dir, f"{project_name}.alma.cycle5.3.ms")

    # 用更匹配的频谱设置进行成像
    tclean(
        vis=ms_file,
        imagename=os.path.join(project_dir, "clean"),
        imsize=[512, 512],  # 确保图像尺寸为512×512
        cell="0.1arcsec",
        specmode="mfs",
        deconvolver="hogbom",
        weighting="natural",
        robust=0.5,
        niter=1000,
        threshold="50uJy",
        interactive=False
    )

    # 导出FITS文件
    image_file = os.path.join(project_dir, "clean.image")
    model_file = os.path.join(project_dir, "clean.model")
    residual_file = os.path.join(project_dir, "clean.residual")

    # 导出clean.image为FITS
    fits_file = os.path.join(project_dir, "clean.fits")
    exportfits(
        imagename=image_file,
        fitsimage=fits_file,
        overwrite=True
    )

    # 导出clean.residual为FITS
    residual_fits = os.path.join(project_dir, "residual.fits")
    exportfits(
        imagename=residual_file,
        fitsimage=residual_fits,
        overwrite=True
    )

    # 导出clean.model为FITS
    model_fits = os.path.join(project_dir, "model.fits")
    exportfits(
        imagename=model_file,
        fitsimage=model_fits,
        overwrite=True
    )

    # 处理clean image
    ia.open(image_file)
    clean_image = ia.getchunk()
    print(f"Clean image shape: {clean_image.shape}")  # 添加调试信息

    if clean_image.ndim == 4:
        clean_image_2d = clean_image[0, 0, :, :]
    else:
        clean_image_2d = clean_image

    print(f"Extracted 2D clean image shape: {clean_image_2d.shape}")  # 添加调试信息
    ia.close()

    # 处理model和residual得到dirty image
    ia.open(model_file)
    model_data = ia.getchunk()
    if model_data.ndim == 4:
        model_data_2d = model_data[0, 0, :, :]
    else:
        model_data_2d = model_data
    print(f"Model data shape: {model_data_2d.shape}")  # 添加调试信息
    ia.close()

    ia.open(residual_file)
    residual_data = ia.getchunk()
    if residual_data.ndim == 4:
        residual_data_2d = residual_data[0, 0, :, :]
    else:
        residual_data_2d = residual_data
    print(f"Residual data shape: {residual_data_2d.shape}")  # 添加调试信息
    ia.close()

    # 计算dirty image
    dirty_image_2d = model_data_2d + residual_data_2d
    print(f"Dirty image shape: {dirty_image_2d.shape}")  # 添加调试信息

    # 确保图像是512×512的，否则进行调整
    if dirty_image_2d.shape[0] != 512 or dirty_image_2d.shape[1] != 512:
        print(f"WARNING: Dirty image shape {dirty_image_2d.shape} is not 512×512, attempting to resize")
        try:
            # 尝试将dirty_image_2d重塑为512×512
            # 这只是一个简单的处理方法，实际情况可能需要更复杂的调整
            if dirty_image_2d.size == 1:
                # 如果只有一个像素，创建一个512×512的空数组
                print("Creating empty 512×512 array filled with the single pixel value")
                value = dirty_image_2d.flatten()[0]
                dirty_image_2d = np.full((512, 512), value)
            else:
                # 其他情况可能需要更复杂的处理
                print("Image resize would require interpolation - not implemented")
        except Exception as e:
            print(f"Error during resizing: {e}")

    # 保存clean image (true) - 使用5个0格式
    np.save(os.path.join(true_dir, f"{sim_idx:05d}.npy"), clean_image_2d)

    # 保存dirty image - 使用5个0格式
    np.save(os.path.join(dirty_dir, f"{sim_idx:05d}.npy"), dirty_image_2d)

# 保存sky_keys.npy
np.save(os.path.join(numpy_dir, "sky_keys.npy"), np.array(all_sky_keys))

# 保存ra_dec.npy - 保存为字典，与样本一致
with open(os.path.join(numpy_dir, "ra_dec.npy"), 'wb') as f:
    pickle.dump(all_ra_dec, f)

# 保存射电源信息
with open(os.path.join(numpy_dir, "sky_sources_snr_extended.npy"), 'wb') as f:
    pickle.dump(all_sources_info, f)

# 恢复原始工作目录
os.chdir(original_dir)

print(f"Simulation complete. Output saved to {base_dir}")
print(f"Data files are in: {data_dir}")
print(f"Clean images (true) are in: {true_dir}")
print(f"Dirty images are in: {dirty_dir}")
print(f"Other NumPy arrays are in: {numpy_dir}")