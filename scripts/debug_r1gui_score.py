#!/usr/bin/env python3
"""Debug r1gui parsing and scoring on one sample."""

import argparse
import json
import sys
from pathlib import Path


def _load_text_arg(value: str) -> str:
    if value.startswith("@"):
        p = Path(value[1:])
        return p.read_text(encoding="utf-8")
    return value


def _load_json_arg(value: str) -> dict:
    if value.startswith("@"):
        p = Path(value[1:])
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--response",
        type=str,
        default=(
            "<thinking>\n"
            "The user has already typed the comment \"哥哥太帅了！鲜活生命力满分！\" in the input field. "
            "The next logical step is to send this comment by clicking the \"发送\" button, "
            "which is located at the bottom right of the screen.\n"
            "</thinking>\n"
            "<answer>click(point='886,547') </answer>"
        ),
        help="Response text. Prefix with @ to read from file.",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=(
            '{"action":"click","gt_bbox":[855.5555555555555,540.8088235294117,928.5714285714286,560.2941176470588],'
            '"input_text":"no input text","gt_params":{"point":"900,550"}}'
        ),
        help="Ground truth JSON string. Prefix with @ to read from file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import verl  # noqa: PLC0415
    from verl.utils.reward_score.r1gui import extract_action, r1gui_compute_score  # noqa: PLC0415

    response = _load_text_arg(args.response)
    ground_truth_obj = _load_json_arg(args.ground_truth)
    ground_truth = json.dumps(ground_truth_obj, ensure_ascii=False)

    print("verl path:", verl.__file__)
    print("response repr:", repr(response))
    print("extract_action:", extract_action(response))
    print("score:", r1gui_compute_score(response, ground_truth))


if __name__ == "__main__":
    main()
