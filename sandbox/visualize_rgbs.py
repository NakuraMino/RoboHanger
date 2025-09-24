import imageio
from pathlib import Path

PATH = Path('/data/minon/RoboHanger_code/outputs/2025-09-23/20-12-06/0/')
VIS_DIR = Path('/data/minon/RoboHanger_code/visualizations/')
if __name__ == "__main__":
    obs_dir = PATH / 'obs'
    topview = True
    nums = [f"{i:02d}" for i in range(50)]
    for num in nums:
        images = []
        if topview:
            files = sorted(list((obs_dir / num / 'color_step_top_view').glob('*.png')))
        else:
            files = sorted(list((obs_dir / num / 'color_step').glob('*.png')))
        for img_path in files:
            images.append(imageio.imread(img_path))
        if topview:
            imageio.mimwrite(VIS_DIR / f'{num}_topview.mp4', images, fps=50)
        else:
            imageio.mimwrite(VIS_DIR / f'{num}_traj.mp4', images, fps=50)
