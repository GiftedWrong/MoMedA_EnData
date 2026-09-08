#!/usr/bin/env python
"""Замер агрегированной скорости переоценки (примеров/мин) для текущих воркеров.

Ждёт начала «свежей» внутренней секции у двух агентов (step < 50), затем мерит
5 минут кумулятивный прогресс с учётом секций (Внутренний=0, usmle=+100).
Устойчив к перезаписи логов (агент сменился) — такие интервалы отбрасываются.
Результат дописывает в throughput_2workers.txt.
"""
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/sgv/Desktop/Dev/AI_Dev/MoMedA_ChData")
BAR_RE = re.compile(r"(Внутренний|usmle)[^\r\n]*?(\d+)/(\d+) \[")


def current_agents():
    out = subprocess.run(["pgrep", "-af", "eval_specialist.py --specialty"],
                         capture_output=True, text=True).stdout
    names = []
    for line in out.splitlines():
        if "bash -c" in line or "pgrep" in line:
            continue
        m = re.search(r"--specialty (\S+)", line)
        if m:
            names.append(m.group(1))
    return names[:2]


def progress(name):
    """кумулятивные завершённые примеры (внутр 0-100, usmle 100-200) или None"""
    try:
        txt = subprocess.run(["tail", "-c", "6000", str(ROOT / f"eval_{name}.log")],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    bars = BAR_RE.findall(txt)
    if not bars:
        return None
    sec, step, _tot = bars[-1]
    return (100 if sec.startswith("usmle") else 0) + int(step)


def wait_fresh_internal(timeout_s=1800):
    """ждём двух агентов на ранней внутренней секции"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        names = current_agents()
        if len(names) == 2:
            ps = [progress(n) for n in names]
            if all(p is not None and 3 <= p < 50 for p in ps):
                return names
        time.sleep(20)
    return None


def main():
    names = wait_fresh_internal()
    if not names:
        print("не дождались свежей внутренней секции за 30 мин", flush=True)
        return
    p1 = {n: progress(n) for n in names}
    t1 = time.time()
    time.sleep(300)
    p2 = {n: progress(n) for n in names}
    t2 = time.time()
    dt = (t2 - t1) / 60
    deltas = [p2[n] - p1[n] for n in names
              if isinstance(p1[n], int) and isinstance(p2[n], int) and p2[n] >= p1[n]]
    if not dt or len(deltas) < 2:
        print(f"интервал нестабилен (смена агентов): {p1} -> {p2}", flush=True)
        return
    rate = sum(deltas) / dt
    with open(ROOT / "throughput_2workers.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M')} агенты {names}: {rate:.1f} примеров/мин "
                f"({dt:.1f} мин, {dict(zip(names, deltas))})\n")
        f.write("базлайн: 3 воркера 15.7/мин, соло ~14.6/мин (внутренняя секция)\n")
    print(f"2 воркера: {rate:.1f} примеров/мин", flush=True)


if __name__ == "__main__":
    main()
