import setuptools
import os
import subprocess

if __name__ == '__main__':
    try:
        setuptools.setup(
            name='robohang',
            version='0.0',
            packages=setuptools.find_packages('src'),
            package_dir={'':'src'},
            python_requires='>=3.8',
            install_requires=[
                "torch",
                "torchvision",
                "numpy",
                "taichi==1.6.0",
                "open3d",
                "tqdm",
                "sapien",
                "lightning",
                "tensorboard",
                "torch_tb_profiler",
                "imageio",
                "pdf2image",
                "pynput",
            ]
        )

        subprocess.run("pip install hydra-core --upgrade", shell=True)

        os.chdir("./external/batch_urdf")
        subprocess.run("pip install -e .", shell=True)
        os.chdir(os.path.dirname(__file__))

    except:
        raise RuntimeError("An error occured during setup.")