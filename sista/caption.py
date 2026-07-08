import unsloth

from abc import ABC
from dataclasses import dataclass
import json_repair
from partial_json_parser import loads

from PIL import Image
import torch

from utils import postprocess_boxes


system_prompt = """
    You are reviewing a drone image that may show the scene of a vehicle accident. Your task is to detect and label all relevant objects in the image, using neutral, evidence-based descriptions. Do not assume every vehicle or person is connected to an accident — only label something as accident-related if there is visible evidence (damage, collision position, debris, blocking traffic, etc.).

    1. Vehicles:
      - Identify and classify all vehicles (cars, trucks, motorcycles), and bicycles only if they are involved in the accident (ignore uninvolved bicycles).
      - Classify each vehicle into exactly one of these three categories:
        * Involved in the accident: visible damage, collision position, debris, or otherwise clearly part of the incident.
        * Emergency or helping vehicle: ambulance, police car, tow truck, fire truck, or similar responding vehicle.
        * Not involved: parked, passing by, or otherwise unconnected to the incident. Use this category by default when there is no visible evidence of involvement.

    2. People:
      - Detect all people present in the scene.
      - Describe their actions and status using neutral, observable terms: injured, standing, sitting, walking, running, helping others, calling for help, waiting, bystander, etc.
      - Do not assume injury or distress unless there is visible evidence; "bystander" or "standing" are valid default labels.

    Output format:
    - Return a valid JSON array with bounding boxes for all detected elements in the form:
      `[{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "detailed description"}, ...]`
    - Example valid response:
      `[{"bbox_2d": [10, 30, 20, 60], "label": "car involved in accident, front-end damage"}, {"bbox_2d": [40, 15, 52, 27], "label": "person injured, sitting"}, {"bbox_2d": [60, 5, 70, 20], "label": "car, not involved, parked"}]`
    - Ensure each label reflects only what is visually evidenced, not assumed.
    - Order the JSON array: accident-related objects first (involved vehicles, emergency vehicles, injured/helping people), then uninvolved objects (parked vehicles, bystanders) last.
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
            model_name=model_name, load_in_4bit=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self.model.eval()

        FastVisionModel.for_inference(self.model)

        self.model.eval()


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

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=2048)

        gen_ids = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        raw_output = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        try:
            data = json_repair.repair_json(raw_output)
            data = loads(data)
            data = postprocess_boxes(data, img)
            
            captions = []
            for item in data:
                if "bbox_2d" in item and "label" in item:
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
        except (Exception, KeyError, TypeError, ValueError) as e:
            # Fallback/Error handling if the model generates malformed JSON
            print(f"Failed to parse model response to JSON: {e}")
            print(f"Raw response was: {raw_output}")
            return []
