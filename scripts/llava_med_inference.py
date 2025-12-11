#!/usr/bin/env python3
"""
LLaVA-Med Inference Wrapper

Simple wrapper around Microsoft's LLaVA-Med for chest X-ray analysis.
"""

import sys
from pathlib import Path

# Add LLaVA-Med to path
sys.path.insert(0, '/Users/ybarbara/www/tcc/LLaVA-Med')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.conversation import conv_templates
from PIL import Image
import torch


class LLaVaMedInference:
    """LLaVA-Med inference wrapper."""
    
    def __init__(self, model_path: str = "microsoft/llava-med-v1.5-mistral-7b"):
        """Initialize LLaVA-Med model."""
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        
        print(f"Loading {self.model_name}...")
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=self.model_name,
                device_map="auto"
            )
        )
        print("✓ Model loaded")
        
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze medical image with LLaVA-Med.
        
        Args:
            image_path: Path to chest X-ray image
            prompt: Medical analysis prompt
            
        Returns:
            Model's analysis text
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        image_tensor = process_images(
            [image], self.image_processor, self.model.config
        )
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        # Prepare conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], f"<image>\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()
        
        # Tokenize
        input_ids = (
            self.tokenizer([prompt_formatted])
            .input_ids.to(self.model.device)
        )
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=512,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
            )
        
        # Decode
        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        
        return outputs


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument(
        "--prompt",
        default="Describe any pathological findings in this chest X-ray.",
        help="Analysis prompt"
    )
    args = parser.parse_args()
    
    # Initialize model
    llava_med = LLaVaMedInference()
    
    # Analyze
    result = llava_med.analyze_image(args.image, args.prompt)
    print(f"\nAnalysis:\n{result}")


if __name__ == "__main__":
    main()
