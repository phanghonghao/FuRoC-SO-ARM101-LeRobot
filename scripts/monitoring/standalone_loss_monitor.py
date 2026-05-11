#!/usr/bin/env python3
"""Standalone ACT training loss monitor for RTX server.

Polls the training log file, parses loss values, detects overfitting.
Usage: python standalone_loss_monitor.py <log_file> [poll_interval_seconds]
"""
import re
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
log = logging.getLogger("act_monitor")

LOSS_RE = [
    re.compile(r"\bloss:([\d.]+)", re.I),        # matches "loss:0.115" (no space)
    re.compile(r"loss:\s+([\d.]+)", re.I),        # matches "loss: 0.115" (with space)
    re.compile(r"train_loss[=:\s]+([\d.]+)", re.I),
]
STEP_RE = [
    re.compile(r"(\d+)/\d+\s"),                   # matches "1234/30000 " from tqdm
    re.compile(r"step[=:\s]+(\d+)", re.I),
]

# State
best_loss = float("inf")
best_step = 0
cur_loss = float("inf")
cur_step = 0
history: list[float] = []
file_pos = 0

# Thresholds
LOSS_RISE_THRESHOLD = 0.20
PLATEAU_WINDOW = 10
PLATEAU_THRESHOLD = 0.01
MIN_STEPS = 500


def parse_loss(line):
    for p in LOSS_RE:
        m = p.search(line)
        if m:
            return float(m.group(1))
    return None


def parse_step(line):
    for p in STEP_RE:
        m = p.search(line)
        if m:
            return int(m.group(1))
    return None


def main():
    global best_loss, best_step, cur_loss, cur_step, history, file_pos

    log_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/logs/train_act_v7.log"
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    log.info("Monitoring: %s (every %ds)", log_file, interval)
    log.info("Thresholds: loss_rise>%.0f%%, plateau_window=%d, plateau_thresh=%.2f",
             LOSS_RISE_THRESHOLD * 100, PLATEAU_WINDOW, PLATEAU_THRESHOLD)

    while True:
        try:
            with open(log_file, "r", errors="replace") as f:
                f.seek(file_pos)
                for line in f:
                    v = parse_loss(line)
                    if v is not None:
                        history.append(v)
                        if v < best_loss:
                            best_loss = v
                            best_step = cur_step
                        cur_loss = v
                    s = parse_step(line)
                    if s is not None:
                        cur_step = max(cur_step, s)
                file_pos = f.tell()
        except Exception as e:
            log.warning("Read error: %s", e)

        if cur_loss < float("inf"):
            # Trend
            trend = "stable"
            if len(history) >= 2:
                recent = history[-5:]
                if len(recent) >= 2:
                    diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
                    avg = sum(diffs) / len(diffs)
                    if abs(avg) / max(recent[-1], 1e-8) > 0.005:
                        trend = "decreasing" if avg < 0 else "increasing"

            # Status
            status = "TRAINING"
            if cur_step > MIN_STEPS and best_loss < float("inf"):
                if cur_loss > best_loss * (1 + LOSS_RISE_THRESHOLD):
                    status = "OVERFITTING"
                elif len(history) >= PLATEAU_WINDOW:
                    window = history[-PLATEAU_WINDOW:]
                    if max(window) > 0:
                        rc = (max(window) - min(window)) / max(window)
                        if rc < PLATEAU_THRESHOLD:
                            status = "CONVERGED"

            log.info(
                "step=%d loss=%.4f best=%.4f@%d trend=%s %s",
                cur_step, cur_loss, best_loss, best_step, trend, status,
            )

            if status == "OVERFITTING":
                log.warning(
                    "!!! OVERFITTING: loss %.4f exceeds best %.4f by >%.0f%%",
                    cur_loss, best_loss, LOSS_RISE_THRESHOLD * 100,
                )
                log.warning("Recommended: stop training, use checkpoint from step %d", best_step)

        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped")
