# (Optional) Passed arguments:
import os
import argparse
from pathlib import Path
from dandi.consts import known_instances

parser = argparse.ArgumentParser(
    description="Train C3PO on spike data from a DANDI NWB dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--cuda-device",
    "--cuda_device",
    dest="cuda_device",
    type=int,
    default=0,
    help="Physical CUDA device to make visible to JAX.",
)

parser.add_argument(
    "--dimensions",
    type=int,
    default=16,
    help="Dimensionality of the C3PO latent and context spaces.",
)

parser.add_argument(
    "--dandiset-id",
    "--dandiset_id",
    dest="dandiset_id",
    type=str,
    default="000138",
    help="DANDI dandiset ID.",
)

parser.add_argument(
    "--dandi-path",
    "--dandi_path",
    dest="dandi_path",
    type=str,
    default=("sub-Jenkins/" "sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"),
    help="Path to the NWB asset within the dandiset.",
)

parser.add_argument(
    "--dandi-instance",
    "--dandi_instance",
    dest="dandi_instance",
    type=str,
    default="dandi",
    choices=sorted(known_instances),
    help="DANDI API instance.",
)

parser.add_argument(
    "--output-dir",
    "--output_dir",
    dest="output_dir",
    type=Path,
    default=None,
    help=(
        "Directory in which to save the trained model. "
        "If omitted, uses c3po_models_<dandiset_id>."
    ),
)

parsed_args = parser.parse_args()
cuda_device, dimensions, dandiset_id, dandi_path, dandi_instance, output_dir = (
    parsed_args.cuda_device,
    parsed_args.dimensions,
    parsed_args.dandiset_id,
    parsed_args.dandi_path,
    parsed_args.dandi_instance,
    parsed_args.output_dir,
)

os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

if output_dir is None:
    output_dir = Path(f"c3po_models_{dandiset_id}")
else:
    output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)


from c3po.utils.dandi import train_from_dandi


def main():
    train_from_dandi(
        dandiset_id=dandiset_id,
        dandi_path=dandi_path,
        dandi_instance=dandi_instance,
        dimensions=dimensions,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
