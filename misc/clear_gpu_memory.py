import subprocess
import os


def main():
    result = subprocess.run(
        "fuser -v /dev/nvidia-uvm", 
        shell=True, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    cmd = "kill -9 " + " ".join(result.stdout.split())
    os.system(cmd)


if __name__ == "__main__":
    main()