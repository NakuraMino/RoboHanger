from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


class SamModel:
    def __init__(
        self, 
        device="cuda", 
        sam_checkpoint: str=None, 
        model_type="vit_h"
    ):
        try:
            import segment_anything as sam_module
            if sam_checkpoint is None:
                sam_checkpoint=os.path.join(os.path.dirname(sam_module.__file__), "..", "ckpt", "sam_vit_h_4b8939.pth")
            self.sam = sam_module.sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            self.predictor = sam_module.SamPredictor(self.sam)
        except ImportError:
            print("Please install segment-anything package by following the instructions in https://github.com/facebookresearch/segment-anything")
            print("Please also put the model checkpoint in the correct path. See the default value in SamModel.__init__()")
        except Exception as e:
            raise e
    
    def predict(self, img: np.ndarray, input_point: np.ndarray, input_label: np.ndarray):
        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return masks, scores, logits


if __name__ == "__main__":
    sam_model = SamModel()
    img = np.array(Image.open("clothes_test.jpeg").resize((1280, 960)))
    input_point = np.array([[350., 130.], [350., 200.]]) * 2
    input_label = np.array([1, 0])
    masks, scores, logits = sam_model.predict(img, input_point, input_label)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(mask)
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()