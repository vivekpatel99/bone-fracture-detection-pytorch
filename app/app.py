# uv run fastapi dev
import io
import pathlib
from typing import Annotated

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from hydra import compose, initialize
from omegaconf import DictConfig
from PIL import Image
from starlette.templating import _TemplateResponse
from torchvision.transforms import v2

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from app.fetch_model_s3 import fetch_model_s3  # noqa: E402

# Define the absolute path to the directory containing this app.py file
# This will be /home/ultron/AI/practice-projects/CV/lung-and-colon-cancer-classification-pytorch/app/
APP_DIR = pathlib.Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
APP = FastAPI()
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))
# Mount static files from the 'static' directory inside the 'app' directory.
# Files will be served under the /app/static URL path.
APP.mount("/app/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def get_cfgs() -> DictConfig:
    with initialize(
        version_base="1.3",
        config_path="../configs",
    ):
        cfg = compose(config_name="eval")
    return cfg


cfg = get_cfgs()
CLASS_NAMES = hydra.utils.instantiate(cfg.model.class_names)


def load_model() -> pl.LightningModule:
    torch.set_grad_enabled(False)
    model_path = fetch_model_s3(cfg)
    input_shape = [3, *hydra.utils.instantiate(cfg.data.train_preprocess_transforms[0].size)]
    cfg.model.net.input_shape = input_shape
    checkpoint = torch.load(model_path, weights_only=False, map_location=DEVICE)  # nosec B614
    # checkpoint = torch.load(cfg.ckpt_path, weights_only=False)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    model.compile_model()
    model.to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


MODEL: pl.LightningModule = load_model()


@APP.get("/")
async def home(request: Request) -> _TemplateResponse:
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@APP.post("/predict")
async def predict(_: Request, file: Annotated[UploadFile, File()] = ...) -> dict[str, str]:
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    transforms = hydra.utils.instantiate(cfg.data.valid_preprocess_transforms)
    transforms += [v2.ToDtype(dtype=torch.float32, scale=True)]
    cmposed_transforms = v2.Compose(transforms)
    img_tensor = cmposed_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        output = MODEL(img_tensor.to(DEVICE))
        pred: int | float = torch.argmax(output, dim=1).item()

    pred_class_name = CLASS_NAMES[pred]

    return {"prediction": pred_class_name}


# if __name__ == "__main__":
#     uvicorn.run(APP, host="0.0.0.0", port=5000)
