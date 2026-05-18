import argparse
import json
import logging
import sys

from dinov3_hot.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dinov3-hot", description="DINOv3-L + UperNet for HOT buildings")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", default="conf/train.yaml", help="Path to YAML config")
        p.add_argument(
            "overrides",
            nargs="*",
            help="OmegaConf dotlist overrides, e.g. data_pct=1 max_epochs=5 run_name=smoke",
        )

    p_train = sub.add_parser("train", help="Train model")
    _common(p_train)

    p_predict = sub.add_parser("predict", help="Sliding-window inference on a GeoTIFF")
    p_predict.add_argument("--ckpt", required=True)
    p_predict.add_argument("--raster", required=True)
    p_predict.add_argument("--out", required=True, help="Output GeoJSON path")
    _common(p_predict)

    p_export = sub.add_parser("export", help="Export checkpoint to ONNX")
    p_export.add_argument("--ckpt", required=True)
    p_export.add_argument("--out", required=True)
    _common(p_export)

    p_evalf = sub.add_parser("eval-fair", help="Run inference on fAIr-models sample tiles")
    p_evalf.add_argument("--ckpt", required=True)
    p_evalf.add_argument("--samples-dir", required=True)
    p_evalf.add_argument("--out-dir", required=True)
    _common(p_evalf)

    p_ft = sub.add_parser("finetune", help="Fine-tune the decoder on a small local labelled set")
    p_ft.add_argument("--ckpt", required=True, help="Pretrained Lightning ckpt to start from")
    p_ft.add_argument("--chips-dir", required=True)
    p_ft.add_argument("--labels-geojson", required=True)
    p_ft.add_argument("--out-dir", required=True)
    p_ft.add_argument("--ft-lr", type=float, default=5e-5)
    p_ft.add_argument("--ft-epochs", type=int, default=15)
    p_ft.add_argument("--val-frac", type=float, default=0.3)
    _common(p_ft)

    return parser


def app() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=list(args.overrides))

    if args.command == "train":
        from dinov3_hot.train import train

        summary = train(cfg)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "predict":
        from dinov3_hot.infer import predict_geotiff

        predict_geotiff(cfg, ckpt_path=args.ckpt, raster_path=args.raster, out_geojson=args.out)
        return 0

    if args.command == "export":
        from dinov3_hot.export import export_onnx

        export_onnx(cfg, ckpt_path=args.ckpt, out_path=args.out)
        return 0

    if args.command == "eval-fair":
        from dinov3_hot.eval_fair import eval_fair_samples

        eval_fair_samples(cfg, ckpt_path=args.ckpt, samples_dir=args.samples_dir, out_dir=args.out_dir)
        return 0

    if args.command == "finetune":
        from dinov3_hot.finetune import finetune

        summary = finetune(
            cfg,
            pretrained_ckpt=args.ckpt,
            chips_dir=args.chips_dir,
            labels_geojson=args.labels_geojson,
            out_dir=args.out_dir,
            val_frac=args.val_frac,
            ft_lr=args.ft_lr,
            ft_epochs=args.ft_epochs,
        )
        print(json.dumps(summary, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(app())
