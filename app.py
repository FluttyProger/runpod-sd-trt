from diffusion import ImageGenerationOptions
from model import DiffusersModel

opts = ImageGenerationOptions(
    prompt="a cow",
    negative_prompt="distorted",
    image=None

)

pipe = DiffusersModel("")

pipe.activate()

for data in pipe(opts):
    if type(data) == tuple:
        pass
    else:
        image = data

results = []
for images, opts in image:
    results.extend(images)


results[0].save("haha.png", format="PNG")
# print(results)