import unsloth

from abc import ABC
from dataclasses import dataclass
import json

from PIL import Image
import torch

from utils import postprocess_boxes


system_prompt = """
    You are an operator supervising a drone operation over a vehicle accident scene.
    Your task is to detect and caption all relevant vehicles and people in the image.

    Only caption and describe vehicles and people involved in an accident, including rescue or helping vehicles.
    The caption and description must focus on the role and involvement vehicles and people have in the accident scene.

    The bounding boxes have already been drawn for all objects in the scene, both involved and not involved in the accident.

    You need to caption exclusively the vehicles objects involved in an accident.
    If a vehicle or person is NOT involved in an accident, explicitly state so.
    The caption must be as short and dense as possible.

    Output format: valid JSON array with bounding boxes for all detected elements in the form: `[{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "detailed description"}, ...]`
"""

user_prompt = """
    Detect and label all relevant vehicles and persons in this frame, only if they are involved in an accident.
"""


@dataclass
class Caption:
    bbox: tuple[float, float, float, float]
    caption: str


class CaptionerQwen3VL:
    def __init__(self, model_name):
        from unsloth import FastVisionModel

        self.model, processor = FastVisionModel.from_pretrained(
            model_name=model_name, load_in_4bit=True
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

        cleaned_output = raw_output.strip()
        
        # Strip markdown code blocks if present
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(cleaned_output)
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
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # Fallback/Error handling if the model generates malformed JSON
            print(f"Failed to parse model response to JSON: {e}")
            print(f"Raw response was: {raw_output}")
            return []
