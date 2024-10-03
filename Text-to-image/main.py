#!pip install --upgrade diffusers transformers -q
"""The command installs or updates the diffusers and transformers libraries in your Python environment, ensuring you have the latest features and improvements for generating images from text and working with AI models"""

from pathlib import Path  # For handling file paths
import tqdm  # For displaying progress bars
import torch  # For deep learning functionalities
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from diffusers import StableDiffusionPipeline  # For generating images using Stable Diffusion
from transformers import pipeline, set_seed  # For NLP tasks and setting random seeds
import matplotlib.pyplot as plt  # For plotting and visualizing images
import cv2  # For image processing
"""The above code imports essential libraries for handling paths, data manipulation, image generation using Stable Diffusion, and visualization. It sets the foundation for building a project that generates images based on text prompts and processes them as needed. """

class CFG:
    device = "cuda"  # Use GPU (CUDA) for processing
    seed = 42  # Random seed for reproducibility of results
    generator = torch.Generator(device).manual_seed(seed)  # Initialize a random generator with the specified seed
    image_gen_steps = 35  # Number of inference steps for generating images
    image_gen_model_id = "stabilityai/stable-diffusion-2"  # Model ID for the Stable Diffusion image generation model
    image_gen_size = (400, 400)  # Size (width, height) of the generated images in pixels
    image_gen_guidance_scale = 9  # Guidance scale to control the quality and adherence of generated images to prompts
    prompt_gen_model_id = "gpt2"  # Model ID for the text generation model (GPT-2)
    prompt_dataset_size = 6  # Size of the dataset used for generating prompts
    prompt_max_length = 12  # Maximum length of the generated prompts in tokens
    """The CFG class acts as a configuration manager for your project, centralizing all the important parameters related to image generation and text prompt handling. This makes it easier to manage and modify settings without changing the core logic of your code."""


    image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
    )
    image_gen_model = image_gen_model.to(CFG.device)
    """The above code initializes the Stable Diffusion model by loading it with specific settings for precision and prompt guidance. It also ensures that the model is set to run on the specified hardware (GPU or CPU), optimizing it for performance during image generation tasks. This setup prepares the model for use in generating images based on text prompts."""


    def generate_image(prompt, model):
        image = model(
            prompt, 
            num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale
        ).images[0]
    
        image = image.resize(CFG.image_gen_size)
            return image

    generate_image("astronaut in space", image_gen_model)
    """The generate_image function encapsulates the process of generating an image from a text prompt using the Stable Diffusion model. It handles all the necessary parameters for the model and ensures that the output image is resized to the desired dimensions. This setup allows for easy image generation by simply calling the function with different prompts."""