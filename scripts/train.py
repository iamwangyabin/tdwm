#!/usr/bin/env python3

from tdwm.training.lewm import build_parser


def main() -> None:
    args = build_parser().parse_args()
    if args.method == "lewm":
        from tdwm.training.lewm import run

        run(args)
        return

    if args.method in {"pldm", "dino_wm", "gcbc", "gcivl", "gciql"}:
        from tdwm.training.baselines import run

        run(args)
        return

    if args.method == "tdmpc2":
        raise RuntimeError(
            "TD-MPC2 remains protocol-gated: it requires reward/Q training and "
            "cannot be mixed into this offline PushT comparison."
        )
    raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
