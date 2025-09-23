import imageio
from pathlib import Path

PATH = Path('/data/minon/RoboHanger_code/outputs/2025-09-21/23-41-37/0/')
VIS_DIR = Path('/data/minon/RoboHanger_code/visualizations/')
if __name__ == "__main__":
    obs_dir = PATH / 'obs'
    nums = [f"{i:01d}" for i in range(50)]
    for num in nums:
        images = []
        for i in range(4600):
            img_path = obs_dir / num / 'color_dense' / f"{i:08d}.png"
            print(img_path)
            images.append(imageio.imread(img_path))
        imageio.mimwrite(VIS_DIR / f'{num}_traj.mp4', images, fps=200)
