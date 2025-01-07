import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderDC
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.cli import tqdm

# torch enable tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-ext", type=str, default="jpg")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--model-name", type=str, default="mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"
    )
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=8)
    return parser.parse_args()


class PortraitDataset(Dataset):
    def __init__(self, data_dir, image_ext="jpg", transform=None, image_size=256):
        self.files = sorted(Path(data_dir).glob(f"*.{image_ext}"))
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = image.resize((self.image_size, self.image_size))
        if self.transform:
            image = self.transform(image)
        return image


def main():
    args = parse_args()

    device = torch.device("cuda")
    dc_ae: AutoencoderDC = (
        AutoencoderDC.from_pretrained(args.model_name, torch_dtype=torch.float32)
        .to(device)
        .eval()
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    dataset = PortraitDataset(args.data_dir, args.image_ext, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    N = len(dataset)
    output = torch.empty(
        N,
        args.num_channels,
        args.output_size,
        args.output_size,
        dtype=torch.float32,
        device="cuda",
    )

    dc_ae = torch.compile(dc_ae)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i, batch in enumerate(tqdm(dataloader, desc="Encoding")):
                batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
                latents = dc_ae.encode(batch).latent
                start_idx = i * dataloader.batch_size
                end_idx = min((i + 1) * dataloader.batch_size, N)
                output[start_idx:end_idx] = latents.data

    # save the output
    torch.save(output, args.output_path)


if __name__ == "__main__":
    main()
