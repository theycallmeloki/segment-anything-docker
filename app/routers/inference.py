import base64
from fastapi import APIRouter, Request, HTTPException
from base64 import b64decode, b64encode
import numpy as np
import torch
import cv2
from ..schemas import SegmentBody
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

router = APIRouter()

@router.post("/segment")
def segment_image(request: Request, body: SegmentBody):
    # Decode the image from base64
    image_data = b64decode(body.image.encode())
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        model_path = "/code/sam_images/sam_vit_l_0b3195.pth"
        model_type = "vit_l"
        device = "cpu"

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)

        results = []
        for ann in masks:
            mask = ann['segmentation']
            success, mask_encoded = cv2.imencode('.png', mask * 255)
            if success:
                mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')  
                results.append({
                    "segmentation": mask_base64,
                    "area": ann['area'],
                    "bbox": ann['bbox'],
                    "predicted_iou": ann['predicted_iou'],
                    "point_coords": ann['point_coords'],
                    "stability_score": ann['stability_score'],
                    "crop_box": ann['crop_box']
                })

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )

        masks2 = mask_generator_2.generate(image)
        results2 = []
        for ann in masks2:
            mask = ann['segmentation']
            success, mask_encoded = cv2.imencode('.png', mask * 255)
            if success:
                mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')  
                results2.append({
                    "segmentation": mask_base64,
                    "area": ann['area'],
                    "bbox": ann['bbox'],
                    "predicted_iou": ann['predicted_iou'],
                    "point_coords": ann['point_coords'],
                    "stability_score": ann['stability_score'],
                    "crop_box": ann['crop_box']
                })

        meta_results = {
            'vit_l_basic': results,
            'vit_l_advanced': results2
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return meta_results
