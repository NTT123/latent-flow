# Latent Flow

This repo provides a simple implementation of a latent flow matching model to generate images. We use the DC-AE model to encode the images into latent space and then train a flow matching model to generate new images in this latent space.

## Train model

We are going to use the [CelebA-HQ (256x256)](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) as our dataset.

```bash
pip install uv
mkdir data
curl -L -o data/celebahq-resized-256x256.zip https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256
unzip data/celebahq-resized-256x256.zip -d data/celebahq
rm data/celebahq-resized-256x256.zip
```

Then we can encode the images into latent space using the `encode_data.py` script.

```bash
uv run encode_data.py --data-dir data/celebahq/celeba_hq_256 --output-path data/latents_256.pt --image-size 256 --model-name mit-han-lab/dc-ae-f32c32-in-1.0-diffusers --num-channels 32 --output-size 8
```

Now we can train the model using the `train.py` script.

```bash
uv run train.py --config config.yaml
```

## Generate images

We can generate images using the `generate.py` script.

```bash
uv run generate.py --config config.yaml
```
