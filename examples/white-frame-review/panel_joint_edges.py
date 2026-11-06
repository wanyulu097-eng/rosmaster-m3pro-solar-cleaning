#!/usr/bin/env python3
"""Offline white-frame edge hypothesis for a physical PV panel image.

Uses median brightness contrast across long Hough segments to distinguish the
white rim/dark glass boundary from glare and robot-body lines. This is a pixel
diagnostic only. A near-image-edge boundary is explicitly blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


def _line_cross(first: np.ndarray, second: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = first.astype(float)
    x3, y3, x4, y4 = second.astype(float)
    a = np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]], dtype=float)
    b = np.array([x3 - x1, y3 - y1], dtype=float)
    t, _ = np.linalg.solve(a, b)
    return [float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))]


def _edge_choices(gray: np.ndarray) -> tuple[dict, dict]:
    height, width = gray.shape
    edges = cv2.Canny(gray, 80, 180)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=70,
        minLineLength=max(80, int(0.25 * height)), maxLineGap=25,
    )
    if segments is None:
        return {side: [] for side in ("top", "right", "bottom", "left")}, {"detected_segments": 0}
    choices: dict[str, list[dict]] = {name: [] for name in ("top", "right", "bottom", "left")}
    offset = max(5, int(round(height / 60)))
    for segment in segments.reshape(-1, 4):
        x1, y1, x2, y2 = segment.astype(float)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        for side in choices:
            if side in ("top", "bottom"):
                if abs(dx) < 1 or abs(dy / dx) > 0.16:
                    continue
            else:
                if abs(dy) < 1 or abs(dx / dy) > 0.35:
                    continue

            fraction = np.linspace(0.15, 0.85, 60)
            xx = x1 + fraction * dx
            yy = y1 + fraction * dy
            if side in ("top", "bottom"):
                outer_y = yy + (offset if side == "bottom" else -offset)
                inner_y = yy - (offset if side == "bottom" else -offset)
                outer_x = inner_x = xx
            else:
                outer_x = xx + (offset if side == "right" else -offset)
                inner_x = xx - (offset if side == "right" else -offset)
                outer_y = inner_y = yy
            valid = (
                (outer_x >= 0) & (outer_x < width) & (inner_x >= 0) & (inner_x < width)
                & (outer_y >= 0) & (outer_y < height) & (inner_y >= 0) & (inner_y < height)
            )
            if np.count_nonzero(valid) < 15:
                continue
            ox, oy = outer_x[valid].astype(int), outer_y[valid].astype(int)
            ix, iy = inner_x[valid].astype(int), inner_y[valid].astype(int)
            contrast = float(np.median(gray[oy, ox].astype(float) - gray[iy, ix].astype(float)))
            if contrast < 50:
                continue
            choices[side].append({
                "segment_px": segment.astype(int).tolist(),
                "median_outer_minus_inner_gray": contrast,
                "length_px": length,
                "score": contrast * length,
            })
    return choices, {"detected_segments": len(segments), "eligible_segments_by_side": {name: len(items) for name, items in choices.items()}}


def _support(start, end, pool):
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        return 0.0
    unit = delta / length
    normal = np.array([-unit[1], unit[0]])
    intervals = []
    for edge in pool:
        points = np.asarray(edge['segment_px'], dtype=float).reshape(2, 2) - start
        if np.max(np.abs(points @ normal)) > 4:
            continue
        lo, hi = sorted(points @ unit)
        lo, hi = max(0.0, float(lo)), min(length, float(hi))
        if hi > lo:
            intervals.append((lo, hi))
    covered, last = 0.0, 0.0
    for lo, hi in sorted(intervals):
        covered += max(0.0, hi - max(lo, last))
        last = max(last, hi)
    return covered / length


def _image_support(gray, start, end, side):
    # Measure polarity on the whole proposed side; Hough fragments are proposals,
    # not evidence for extrapolated gaps. Keep the existing contrast/offset.
    height, width = gray.shape
    points = start + np.linspace(.02, .98, 120)[:, None] * (end - start)
    offset = max(5, int(round(height / 60)))
    shift = {'top': (0, -offset), 'bottom': (0, offset),
             'left': (-offset, 0), 'right': (offset, 0)}[side]
    outer, inner = points + shift, points - shift
    valid = (np.all(outer >= 0, axis=1) & np.all(inner >= 0, axis=1)
             & (outer[:, 0] < width) & (inner[:, 0] < width)
             & (outer[:, 1] < height) & (inner[:, 1] < height))
    if not np.all(valid):
        return 0.0
    ox, oy = outer.astype(int).T
    ix, iy = inner.astype(int).T
    contrast = gray[oy, ox].astype(float) - gray[iy, ix].astype(float)
    return float(np.mean(contrast >= 50))


def _best_edges(gray):
    pools, diagnostics = _edge_choices(gray)
    height, width = gray.shape
    sides = ('top', 'right', 'bottom', 'left')
    # ponytail: bounded enumeration for an offline candidate; no realtime claim.
    combinations = int(np.prod([len(pools[side]) for side in sides]))
    diagnostics.update(combinations=combinations, closed_supported_groups=0,
                       support_threshold=0.65, support_metric="full_side_pixel_polarity_fraction", support_contrast=50)
    if combinations > 25000:
        diagnostics['group_blocker'] = 'too many edge combinations for bounded search'
        return {}, diagnostics
    best = None
    for group in itertools.product(*(pools[side] for side in sides)):
        lines = [np.asarray(item['segment_px'], dtype=float) for item in group]
        try:
            quad = np.asarray([_line_cross(lines[0], lines[3]), _line_cross(lines[0], lines[1]),
                               _line_cross(lines[2], lines[1]), _line_cross(lines[2], lines[3])])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(quad)):
            continue
        tl, tr, br, bl = quad
        if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
            continue
        if np.any(quad[:, 0] < 8) or np.any(quad[:, 0] > width - 9) or np.any(quad[:, 1] < 8) or np.any(quad[:, 1] > height - 9):
            continue
        contour = quad.astype(np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < .04 * width * height:
            continue
        support = [_image_support(gray, quad[i], quad[(i + 1) % 4], side) for i, side in enumerate(sides)]
        if min(support) < .65:
            continue
        diagnostics['closed_supported_groups'] += 1
        score = (min(support), sum(support), sum(item['score'] for item in group))
        if best is None or score > best[0]:
            best = score, group, support
    if best is None:
        diagnostics['group_blocker'] = 'no ordered closed quadrilateral with four supported edges'
        return {}, diagnostics
    diagnostics['selected_support_fraction'] = dict(zip(sides, best[2]))
    lines = [np.asarray(item['segment_px'], dtype=float) for item in best[1]]
    quad = np.asarray([_line_cross(lines[0], lines[3]), _line_cross(lines[0], lines[1]),
                       _line_cross(lines[2], lines[1]), _line_cross(lines[2], lines[3])])
    diagnostics['selected_hough_support_fraction'] = {side: _support(quad[i], quad[(i+1)%4], pools[side]) for i, side in enumerate(sides)}
    return dict(zip(sides, best[1])), diagnostics


def analyze_color(color_bytes: bytes) -> dict:
    """Return the existing pixel hypothesis without CLI, disk or ROS effects."""
    image = cv2.imdecode(np.frombuffer(color_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("color input cannot be decoded")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    selected, diagnostics = _best_edges(gray)
    missing = [side for side in ("top", "right", "bottom", "left") if side not in selected]
    blockers = [f"no bright-rim/dark-glass contrast line for {side}" for side in missing]
    quad = None
    if not missing:
        segments = {side: np.asarray(value["segment_px"], dtype=float) for side, value in selected.items()}
        try:
            quad = np.array([
                _line_cross(segments["top"], segments["left"]),
                _line_cross(segments["top"], segments["right"]),
                _line_cross(segments["bottom"], segments["right"]),
                _line_cross(segments["bottom"], segments["left"]),
            ])
            if not np.all(np.isfinite(quad)):
                blockers.append("line intersections are not finite")
            if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= width) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= height):
                blockers.append("a proposed corner is outside the image")
            margin = float(np.min(np.column_stack((quad[:, 0], width - 1 - quad[:, 0], quad[:, 1], height - 1 - quad[:, 1]))))
            if margin < 8.0:
                blockers.append("white-frame hypothesis is within 8 pixels of the image edge")
            if cv2.contourArea(quad.astype(np.float32)) < 0.04 * width * height:
                blockers.append("white-frame hypothesis is too small")
        except np.linalg.LinAlgError:
            blockers.append("selected lines do not intersect reliably")

    return {
        "schema": "pv_m3pro_panel_joint_edges_v1",
        "color_sha256": hashlib.sha256(color_bytes).hexdigest(),
        "image_size_px": [width, height],
        "decision": "WHITE_FRAME_HYPOTHESIS_BLOCKED" if blockers else "WHITE_FRAME_PIXEL_HYPOTHESIS_ONLY",
        "qualified_full_panel": False,
        "metric_bound": False,
        "coverage_planning_allowed": False,
        "motion_allowed": False,
        "published_ros_topic": False,
        "selected_edges": selected,
        "quad_px_tl_tr_br_bl": quad.tolist() if quad is not None and np.all(np.isfinite(quad)) else None,
        "diagnostics": diagnostics,
        "blockers": blockers,
        "limitation": "Color line contrast and a four-corner intersection are not independent proof of full visible panel or metric geometry.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    color_bytes = args.color.read_bytes()
    result = analyze_color(color_bytes)
    image = cv2.imdecode(np.frombuffer(color_bytes, np.uint8), cv2.IMREAD_COLOR)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    for side, value in result["selected_edges"].items():
        x1, y1, x2, y2 = value["segment_px"]
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(overlay, side, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
    quad = result["quad_px_tl_tr_br_bl"]
    if quad is not None:
        cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 255, 255), 2)
    overlay_path = output / "white_frame_review_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"could not save {overlay_path}")
    result["overlay_file"] = str(overlay_path)
    (output / "white_frame_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("decision", "quad_px_tl_tr_br_bl", "blockers", "diagnostics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

