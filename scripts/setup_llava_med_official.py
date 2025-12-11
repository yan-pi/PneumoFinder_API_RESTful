#!/usr/bin/env python3
"""
LLaVA-Med Official Setup Script

Uses the official Microsoft LLaVA-Med repository and codebase.
This bypasses transformers architecture issues by using their custom implementation.

Steps:
1. Install LLaVA-Med package dependencies
2. Download model from HuggingFace (no merging needed for v1.5)
3. Test inference with their serving API

Optimized for M4 Pro 24GB with MPS
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
LLAVA_MED_DIR = Path("/Users/ybarbara/www/tcc/LLaVA-Med")
MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"


def check_repo_exists() -> bool:
    """Check if LLaVA-Med repo is cloned."""
    return LLAVA_MED_DIR.exists() and (LLAVA_MED_DIR / "llava").exists()


def install_dependencies() -> bool:
    """Install LLaVA-Med package dependencies."""
    logger.info("Installing LLaVA-Med dependencies...")

    try:
        # Install in editable mode
        result = subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=LLAVA_MED_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Installation failed: {result.stderr}")
            return False

        logger.info("✓ Dependencies installed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Installation timed out")
        return False
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        return False


def test_model_loading() -> bool:
    """Test if model can be loaded with LLaVA-Med code."""
    logger.info("Testing model loading...")

    test_script = """
import sys
sys.path.insert(0, '/Users/ybarbara/www/tcc/LLaVA-Med')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = 'microsoft/llava-med-v1.5-mistral-7b'
model_name = get_model_name_from_path(model_path)

print(f'Loading {model_name}...')
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device_map='auto'
)

print('✓ Model loaded successfully')
print(f'Context length: {context_len}')
"""

    try:
        result = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"Model loading failed: {result.stderr}")
            return False

        logger.info(result.stdout)
        return True

    except subprocess.TimeoutExpired:
        logger.error("Model loading timed out")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def create_inference_wrapper() -> None:
    """Create a simple Python wrapper for LLaVA-Med inference."""
    wrapper_path = Path(__file__).parent / "llava_med_inference.py"

    wrapper_code = '''#!/usr/bin/env python3
"""
LLaVA-Med Inference Wrapper

Simple wrapper around Microsoft\'s LLaVA-Med for chest X-ray analysis.
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
            Model\'s analysis text
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
        conv.append_message(conv.roles[0], f"<image>\\n{prompt}")
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
    print(f"\\nAnalysis:\\n{result}")


if __name__ == "__main__":
    main()
'''

    wrapper_path.write_text(wrapper_code)
    wrapper_path.chmod(0o755)
    logger.info(f"✓ Created inference wrapper: {wrapper_path}")


def main() -> None:
    """Main setup script."""
    parser = argparse.ArgumentParser(description="Setup LLaVA-Med using official Microsoft repo")
    parser.add_argument("--skip-test", action="store_true", help="Skip model loading test")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║        LLaVA-Med Official Repository Setup               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Check repo
    if not check_repo_exists():
        logger.error(f"LLaVA-Med repo not found at {LLAVA_MED_DIR}")
        logger.error("Please clone first:")
        logger.error("  cd /Users/ybarbara/www/tcc")
        logger.error("  git clone https://github.com/microsoft/LLaVA-Med.git")
        sys.exit(1)

    logger.info(f"✓ Found LLaVA-Med repo at {LLAVA_MED_DIR}")

    # Install dependencies
    if not install_dependencies():
        logger.error("Dependency installation failed")
        sys.exit(1)

    # Create wrapper
    create_inference_wrapper()

    # Test model
    if not args.skip_test:
        logger.info("\nTesting model loading (this may take 5-10 minutes)...")
        if not test_model_loading():
            logger.warning("Model loading test failed, but setup completed")
            logger.warning("You can test manually later")

    logger.info("\n✓ LLaVA-Med setup complete!")
    logger.info("\nUsage:")
    logger.info("  python scripts/llava_med_inference.py \\")
    logger.info("    --image imgs/person75_bacteria_365.jpeg \\")
    logger.info("    --prompt 'Describe pathological findings'")


if __name__ == "__main__":
    main()
