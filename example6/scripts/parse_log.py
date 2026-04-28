"""학습 로그 파일을 파싱해서 최신 메트릭 JSON을 stdout으로 출력.

autonomous loop가 학습 stdout을 파일로 저장한 뒤 이 스크립트로 마지막
[env0] [dist] [rate] [rew] 섹션을 읽고 판단에 사용.

사용:
    python source/example6/scripts/parse_log.py logs/iter_001.log
"""
import json
import re
import sys
from pathlib import Path


SECTION_KEYS = {
    "env0": re.compile(
        r"\[env0\]\s+obj=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+\|\s+"
        r"grip=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+\|\s+"
        r"tgt=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+\|\s+"
        r"upright=([-\d.]+)\s+\|\s+grip_close=([-\d.]+)"
    ),
    "joints": re.compile(
        r"\[env0_joints\]\s+j1=([-+0-9.]+)\s+j2=([-+0-9.]+)\s+"
        r"j3=([-+0-9.]+)\s+j4=([-+0-9.]+)\s+j5=([-+0-9.]+)\s+j6=([-+0-9.]+)"
    ),
}

# 단순 key=value 패턴으로 dist / rate / rew 뽑기
GROUP_PATTERN = re.compile(r"\[(dist|rate|rew)\]\s+(.+)$")
KV_PATTERN = re.compile(r"([a-z0-9_]+)=([-+\d.eE]+)")

SESSION_PATTERN = re.compile(
    r"session_step=\s*([\d,]+)\s+\|\s+total_step=\s*([\d,]+)\s+\|\s+reward=\s*([-\d.]+)"
)


def parse_log(path: str) -> dict:
    text = Path(path).read_text(errors="ignore")
    lines = text.splitlines()

    metrics: dict = {
        "env0": {},
        "dist": {},
        "rate": {},
        "rew": {},
        "session_step": 0,
        "total_step": 0,
        "ep_reward": 0.0,
        "parsed_lines": 0,
        "last_print_found": False,
    }

    # 가장 최근 blocks 사용 (파일 끝쪽 우선)
    # 라인들을 역순으로 보면서 처음 만나는 env0 블록을 사용
    last_env0_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "[env0]" in lines[i] and "obj=" in lines[i]:
            last_env0_idx = i
            break

    if last_env0_idx is None:
        return metrics

    # last_env0_idx에서 ~20줄 범위 스캔
    start = max(0, last_env0_idx - 10)
    end = min(len(lines), last_env0_idx + 10)
    block = lines[start:end]

    for line in block:
        m = SECTION_KEYS["env0"].search(line)
        if m:
            vals = [float(x) for x in m.groups()]
            metrics["env0"].update({
                "obj_x": vals[0], "obj_y": vals[1], "obj_z": vals[2],
                "grip_x": vals[3], "grip_y": vals[4], "grip_z": vals[5],
                "tgt_x": vals[6], "tgt_y": vals[7], "tgt_z": vals[8],
                "upright": vals[9], "grip_close": vals[10],
            })
            continue

        m = SECTION_KEYS["joints"].search(line)
        if m:
            vals = [float(x) for x in m.groups()]
            metrics["env0"].update({
                f"j{i+1}": vals[i] for i in range(6)
            })
            continue

        m = GROUP_PATTERN.search(line)
        if m:
            group = m.group(1)
            kvs = KV_PATTERN.findall(m.group(2))
            for k, v in kvs:
                try:
                    metrics[group][k] = float(v)
                except ValueError:
                    pass
            continue

        m = SESSION_PATTERN.search(line)
        if m:
            metrics["session_step"] = int(m.group(1).replace(",", ""))
            metrics["total_step"] = int(m.group(2).replace(",", ""))
            try:
                metrics["ep_reward"] = float(m.group(3))
            except ValueError:
                pass
            metrics["last_print_found"] = True

    metrics["parsed_lines"] = len(block)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_log.py <log_file>", file=sys.stderr)
        sys.exit(1)
    result = parse_log(sys.argv[1])
    print(json.dumps(result, indent=2))
