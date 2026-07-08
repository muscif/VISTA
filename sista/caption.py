import unsloth

from abc import ABC
from dataclasses import dataclass
import json_repair

from PIL import Image
import torch

from utils import postprocess_boxes


system_prompt = """
    You are an operator supervising a drone operation over an accident scene. Your task is to detect and label all relevant objects in the images. Focus on the following:

    1. Vehicles:
      - Identify and classify all vehicles, including cars, trucks, motorcycles, bicycles only if they are involved in the accident, ignore the rest.
      - Distinguish between:
        * Vehicles involved in the accident
        * Emergency or helping vehicles

    2. People:
      - Detect all people present in the scene.
      - Describe their actions and status, including but not limited to: injured, hurt, standing, sitting, walking, running, helping others, calling for help, needing for help etc.
      - Include this information in the label.

    Output format:
    - Return a valid JSON array with bounding boxes for all detected elements in the form:
      `[{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "detailed description"}, ...]`
    - Example valid response:
      `[{"bbox_2d": [10, 30, 20, 60], "label": "car involved in accident"}, {"bbox_2d": [40, 15, 52, 27], "label": "person injured, sitting"}]`
    - Ensure each object is labeled with a precise description reflecting its type and status.
"""

user_prompt = """
    Detect and label all relevant vehicles and persons in this frame.
"""


@dataclass
class Caption:
    bbox: tuple[float, float, float, float]
    caption: str


class CaptionerQwen3VL:
    def __init__(self, model_name):
        from unsloth import FastVisionModel

        self.model, processor = FastVisionModel.from_pretrained(
            model_name=model_name, load_in_4bit=True,
        )

        FastVisionModel.for_inference(self.model)

        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
    
    def caption(self, img: Image.Image):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": self.user_prompt},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=8192)

        gen_ids = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        raw_output = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        try:
            data = json_repair.loads(raw_output)
            data = postprocess_boxes(data, img)
            
            captions = []
            for item in data:
                # Convert the list of 4 floats into a tuple as required by Caption dataclass
                bbox_tuple = tuple(item["bbox_2d"])
                
                if len(bbox_tuple) == 4:
                    captions.append(
                        Caption(
                            bbox=bbox_tuple,  # type: ignore
                            caption=item["label"]
                        )
                    )
            return captions
        except (json_repair.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # Fallback/Error handling if the model generates malformed JSON
            print(f"Failed to parse model response to JSON: {e}")
            print(f"Raw response was: {raw_output}")
            return []
