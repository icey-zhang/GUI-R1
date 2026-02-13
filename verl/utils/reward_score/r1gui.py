import ast
import json
import math
import re
from typing import Any, Dict, Optional, Tuple


OUTER_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
POINT_TAG_RE = re.compile(r"<point>\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*</point>")
POINT_CSV_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*$")

VALID_ACTIONS = {
    "click",
    "long_press",
    "type",
    "swipe",
    "open_app",
    "drag",
    "press_home",
    "press_back",
    "wait",
    "finished",
    "call_user",
    "back_information",
    # legacy aliases
    "scroll",
    "complete",
    "select",
    "enter",
    "close/delete",
}


def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str = str(predicted_str).replace("[", "").replace("]", "")
    ground_truth_str = str(ground_truth_str).replace("[", "").replace("]", "")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def _extract_answer_content(predict_str: str) -> Optional[str]:
    if not re.fullmatch(OUTER_PATTERN, predict_str):
        return None
    answer_match = re.search(ANSWER_PATTERN, predict_str)
    if not answer_match:
        return None
    return answer_match.group(1).strip()


def _canonical_action(action: str) -> str:
    action = str(action).strip().lower()
    if action == "complete":
        return "finished"
    if action == "scroll":
        return "swipe"
    if action == "select":
        return "type"
    return action


def _normalize_direction(direction: str) -> str:
    direction = str(direction).strip().lower()
    alias = {"top": "up", "bottom": "down"}
    return alias.get(direction, direction)


def _parse_point(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    if isinstance(value, str):
        value = value.strip()
        m = POINT_TAG_RE.search(value)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = POINT_CSV_RE.match(value)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def _point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _parse_call_action(answer_content: str) -> Optional[Dict[str, Any]]:
    candidate = answer_content.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        inner = candidate[1:-1].strip()
        if inner and "{" not in inner:
            candidate = inner
    if candidate.endswith(";"):
        candidate = candidate[:-1].strip()

    try:
        expr = ast.parse(candidate, mode="eval").body
    except Exception:
        return None

    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return None

    action = expr.func.id
    params: Dict[str, str] = {}
    for kw in expr.keywords:
        if kw.arg is None:
            return None
        value_node = kw.value
        if isinstance(value_node, ast.Constant):
            params[kw.arg] = str(value_node.value)
            continue
        if (
            isinstance(value_node, ast.UnaryOp)
            and isinstance(value_node.op, ast.USub)
            and isinstance(value_node.operand, ast.Constant)
            and isinstance(value_node.operand.value, (int, float))
        ):
            params[kw.arg] = str(-value_node.operand.value)
            continue
        return None

    return {"action": str(action), "params": params}


def _parse_legacy_action(answer_content: str) -> Optional[Dict[str, Any]]:
    try:
        obj = ast.literal_eval(answer_content)
    except Exception:
        return None

    if isinstance(obj, list):
        if not obj:
            return None
        obj = obj[0]
    if not isinstance(obj, dict):
        return None

    action = str(obj.get("action", "")).strip()
    if not action:
        return None

    point = obj.get("point")
    input_text = str(obj.get("input_text", ""))
    params: Dict[str, str] = {}

    if isinstance(point, (list, tuple)) and len(point) >= 2:
        params["point"] = f"{int(round(float(point[0])))},{int(round(float(point[1])))}"

    canon_action = _canonical_action(action)
    if canon_action in {"type"}:
        params["content"] = input_text
    elif canon_action == "open_app":
        params["app_name"] = input_text
    elif canon_action == "wait":
        params["t"] = input_text
    elif canon_action == "swipe":
        if input_text:
            params["direction"] = input_text
    elif canon_action in {"finished", "call_user", "back_information"}:
        params["content"] = input_text

    return {"action": action, "params": params}


def _parse_predicted_action(predict_str: str) -> Optional[Dict[str, Any]]:
    answer_content = _extract_answer_content(predict_str)
    if answer_content is None:
        return None

    parsed = _parse_call_action(answer_content)
    if parsed is not None:
        return parsed
    return _parse_legacy_action(answer_content)


def _normalize_gt_params(raw_gt_params: Any) -> Dict[str, str]:
    if isinstance(raw_gt_params, dict):
        return {str(k): str(v) for k, v in raw_gt_params.items()}
    if isinstance(raw_gt_params, str) and raw_gt_params.strip():
        try:
            obj = json.loads(raw_gt_params)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except Exception:
            return {}
    return {}


def _normalize_bbox(raw_bbox: Any) -> list[float]:
    if not isinstance(raw_bbox, list):
        return []
    if len(raw_bbox) not in (2, 4):
        return []
    try:
        return [float(v) for v in raw_bbox]
    except Exception:
        return []


def _derive_gt_params_from_legacy(gt_action: str, gt_bbox: list[float], gt_input_text: str) -> Dict[str, str]:
    action = _canonical_action(gt_action)
    if action in {"click", "long_press"}:
        if len(gt_bbox) == 2:
            return {"point": f"{int(round(gt_bbox[0]))},{int(round(gt_bbox[1]))}"}
        if len(gt_bbox) == 4:
            cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
            cy = (gt_bbox[1] + gt_bbox[3]) / 2.0
            return {"point": f"{int(round(cx))},{int(round(cy))}"}
    if action == "swipe":
        if gt_input_text:
            return {"direction": gt_input_text}
    if action == "type":
        return {"content": gt_input_text}
    if action == "open_app":
        return {"app_name": gt_input_text}
    if action == "wait":
        return {"t": gt_input_text}
    if action in {"finished", "call_user", "back_information"}:
        return {"content": gt_input_text}
    return {}


def _is_format_valid(parsed_action: Dict[str, Any]) -> bool:
    action = _canonical_action(parsed_action.get("action", ""))
    params = parsed_action.get("params", {})
    if not isinstance(params, dict):
        return False

    if action not in VALID_ACTIONS and action not in {_canonical_action(a) for a in VALID_ACTIONS}:
        return False

    if action in {"click", "long_press"}:
        return _parse_point(params.get("point")) is not None

    if action == "swipe":
        start_p = _parse_point(params.get("start_point"))
        end_p = _parse_point(params.get("end_point"))
        if start_p is not None and end_p is not None:
            velocity = params.get("velocity", "600")
            v = _parse_float(velocity)
            return v is None or v > 0
        direction = _normalize_direction(params.get("direction", ""))
        return direction in {"up", "down", "left", "right"}

    if action == "drag":
        return _parse_point(params.get("start_point")) is not None and _parse_point(params.get("end_point")) is not None

    if action == "type":
        return "content" in params

    if action == "open_app":
        return "app_name" in params

    if action == "wait":
        t = _parse_float(params.get("t", None))
        return t is not None and 0 < t < 10

    if action in {"finished", "call_user", "back_information"}:
        return "content" in params

    if action in {"press_home", "press_back", "enter", "close/delete"}:
        return True

    return False


def extract_action(content):
    parsed = _parse_predicted_action(content)
    if parsed is None:
        return "no action"
    return str(parsed.get("action", "no action"))


def extract_input_text(content):
    parsed = _parse_predicted_action(content)
    if parsed is None:
        return "no input text"
    action = _canonical_action(parsed.get("action", ""))
    params = parsed.get("params", {})
    if action in {"type", "finished", "call_user", "back_information"}:
        return str(params.get("content", "no input text"))
    if action == "open_app":
        return str(params.get("app_name", "no input text"))
    if action == "wait":
        return str(params.get("t", "no input text"))
    if action == "swipe":
        return str(params.get("direction", "no input text"))
    return "no input text"


def extract_coord(content):
    parsed = _parse_predicted_action(content)
    if parsed is None:
        return [0, 0, 0, 0], False
    action = _canonical_action(parsed.get("action", ""))
    if action not in {"click", "long_press"}:
        return [0, 0, 0, 0], False
    point = _parse_point(parsed.get("params", {}).get("point"))
    if point is None:
        return [0, 0, 0, 0], False
    return [int(round(point[0])), int(round(point[1]))], True


def r1gui_format_reward(predict_str: str) -> float:
    parsed = _parse_predicted_action(predict_str)
    if parsed is None:
        return 0.0
    if not _is_format_valid(parsed):
        return 0.0
    return 1.0


def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        pred = _parse_predicted_action(predict_str)
        if pred is None:
            return 0.0

        gt = json.loads(ground_truth)
        gt_action = _canonical_action(gt.get("action", ""))
        gt_bbox = _normalize_bbox(gt.get("gt_bbox", []))
        gt_input_text = str(gt.get("input_text", ""))
        gt_params = _normalize_gt_params(gt.get("gt_params", {}))
        if not gt_params:
            gt_params = _derive_gt_params_from_legacy(gt_action, gt_bbox, gt_input_text)

        pred_action = _canonical_action(pred.get("action", ""))
        pred_params = {str(k): str(v) for k, v in (pred.get("params", {}) or {}).items()}

        if pred_action != gt_action:
            return 0.0

        if pred_action in {"click", "long_press"}:
            pred_point = _parse_point(pred_params.get("point"))
            if pred_point is None:
                return 0.0

            if len(gt_bbox) == 4:
                x1, y1, x2, y2 = gt_bbox
                if (x1 <= pred_point[0] <= x2) and (y1 <= pred_point[1] <= y2):
                    return 1.0

            gt_point = None
            if len(gt_bbox) == 2:
                gt_point = (gt_bbox[0], gt_bbox[1])
            elif len(gt_bbox) == 4:
                gt_point = ((gt_bbox[0] + gt_bbox[2]) / 2.0, (gt_bbox[1] + gt_bbox[3]) / 2.0)
            if gt_point is None:
                gt_point = _parse_point(gt_params.get("point"))
            if gt_point is None:
                return 0.0
            return 1.0 if _point_distance(pred_point, gt_point) < 140 else 0.0

        if pred_action in {"swipe", "drag"}:
            pred_start = _parse_point(pred_params.get("start_point"))
            pred_end = _parse_point(pred_params.get("end_point"))
            gt_start = _parse_point(gt_params.get("start_point"))
            gt_end = _parse_point(gt_params.get("end_point"))

            if pred_start and pred_end and gt_start and gt_end:
                if _point_distance(pred_start, gt_start) < 180 and _point_distance(pred_end, gt_end) < 180:
                    return 1.0
                return 0.0

            pred_direction = _normalize_direction(pred_params.get("direction", ""))
            gt_direction = _normalize_direction(gt_params.get("direction", gt_input_text))
            if pred_direction and gt_direction:
                return 1.0 if pred_direction == gt_direction else 0.0
            return 0.0

        if pred_action == "type":
            pred_content = pred_params.get("content", "")
            gt_content = gt_params.get("content", gt_input_text)
            return 1.0 if calculate_f1_score(pred_content, gt_content) >= 0.5 else 0.0

        if pred_action == "open_app":
            pred_app = pred_params.get("app_name", "")
            gt_app = gt_params.get("app_name", gt_input_text)
            return 1.0 if calculate_f1_score(pred_app, gt_app) >= 0.5 else 0.0

        if pred_action == "wait":
            pred_t = _parse_float(pred_params.get("t", None))
            gt_t = _parse_float(gt_params.get("t", gt_input_text))
            if pred_t is None or gt_t is None:
                return 0.0
            return 1.0 if abs(pred_t - gt_t) <= 0.5 else 0.0

        if pred_action in {"finished", "call_user", "back_information"}:
            pred_content = pred_params.get("content", "")
            gt_content = gt_params.get("content", gt_input_text)
            return 1.0 if calculate_f1_score(pred_content, gt_content) >= 0.5 else 0.0

        if pred_action in {"press_home", "press_back", "enter", "close/delete"}:
            return 1.0

        return 0.0
    except Exception:
        return 0.0


def r1gui_compute_score(predict_str: str, ground_truth: str):
    format_score = r1gui_format_reward(predict_str)
    accuracy = r1gui_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.8 * accuracy + 0.2 * format_score,
        "format": format_score,
        "accuracy": accuracy,
    }
