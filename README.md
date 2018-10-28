# cougar-or-not

My first attempt at a machine learning API, using a pre-calculated model trained
using iNaturalist data.

The model is `usa-inaturalist-cats.pth` - an 83MB file.

`cougar.py` is a very tiny [Starlette](https://www.starlette.io/) API server
which simply accepts file image uploads and runs them against the pre-calculated
[fastai](https://github.com/fastai/fastai) model.

The `Dockerfile` means the entire thing can be deployed to
[Zeit Now](https://zeit.co/now) or any other container hosting service.
