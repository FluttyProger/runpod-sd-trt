import base64
from io import BytesIO

import runpod
from PIL import Image
from diffusion import ImageGenerationOptions
from model import DiffusersModel

print('run handler')

pipe = DiffusersModel("")

pipe.activate()

# pipe.pipe.enable_xformers_memory_efficient_attention()
#
# pipe.pipe.enable_attention_slicing()

def handler(event):
    model_inputs = event['input']
    prompt = model_inputs.get('prompt', "Pepe")
    height = model_inputs.get('height', 512)
    negative = model_inputs.get('negative_prompt', "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation")
    width = model_inputs.get('width', 512)
    steps = model_inputs.get('steps', 36)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    seed = model_inputs.get('seed', -1)
    sampler = model_inputs.get('sampler', "dpm++")
    strength = model_inputs.get('strength', 1.0)
    initimg = model_inputs.get('image', None)

    opts = ImageGenerationOptions(image=initimg, strength=strength, scheduler_id=sampler, prompt=prompt, height=height, negative_prompt=negative, width=width, num_inference_steps=steps, guidance_scale=guidance_scale, seed=seed)

    if initimg is not None:
        opts.image = Image.open(BytesIO(base64.b64decode(initimg)))

    for data in pipe(opts):
        if type(data) == tuple:
            pass
        else:
            image = data

    results = []
    for images, opts in image:
        results.extend(images)

    # do the things
    image = results[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return {'image_base64': image_base64}


# handler({"input":{"prompt":"Image"}})
runpod.serverless.start({"handler": handler})
