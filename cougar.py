from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from fastai.vision import (
    ImageDataBunch,
    ConvLearner,
    open_image,
    get_transforms,
    models,
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn

app = Starlette()

cat_images_path = Path("/tmp")
cat_fnames = [
    "/{}_1.jpg".format(c)
    for c in [
        "Bobcat",
        "Mountain-Lion",
        "Domestic-Cat",
        "Western-Bobcat",
        "Canada-Lynx",
        "North-American-Mountain-Lion",
        "Eastern-Bobcat",
        "Central-American-Ocelot",
        "Ocelot",
        "Jaguar",
    ]
]
cat_data = ImageDataBunch.from_name_re(
    cat_images_path,
    cat_fnames,
    r"/([^/]+)_\d+.jpg$",
    ds_tfms=get_transforms(),
    size=224,
)
cat_learner = ConvLearner(cat_data, models.resnet34)
cat_learner.model.load_state_dict(
    torch.load("usa-inaturalist-cats.pth", map_location="cpu")
)


@app.route("/upload", methods=["POST"])
async def homepage(request):
    data = await request.form()
    bytes = await (data["file"].read())
    img = open_image(BytesIO(bytes))
    losses = img.predict(cat_learner)
    prediction = cat_learner.data.classes[losses.argmax()]
    return JSONResponse({"prediction": prediction})


@app.route("/form")
def form(request):
    return HTMLResponse(
        """
    <form action="/upload" method="post" enctype="multipart/form-data">
    Select image to upload:
    <input type="file" name="file">
    <input type="submit" value="Upload Image">
    </form>
    """
    )


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
