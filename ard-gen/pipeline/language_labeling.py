"""ARD-Gen 5단계: force/torque profile로 자연어 지시문을 생성한다.

## 템플릿 기반으로 구현한 이유 (Claude API 대신)

지금은 파이프라인이 0->1->2->4->5까지 끝까지 도는지를 확인하는 단계다 --
"지시문이 얼마나 자연스러운가"가 아니라 "language 필드가 실제로 채워져서
끝까지 나오는가"가 검증 목표라서, 템플릿으로도 목적에 충분하다. 게다가:

1. 오프라인/재현 가능해야 한다 -- API 키/네트워크/비용 없이 몇 번을
   돌려도 항상 같은 결과가 나와야 지금 이 검증 단계에 맞다.
2. PIPELINE.md의 설계 원칙 2("계산 비용이 싼 방법부터 쓴다")와 같은 맥락 --
   지금 단계에서 LLM 호출은 과한 비용이다.
3. 지금 검증 태스크가 peg-in-hole 하나뿐이라 표현할 정보가 "삽입 강도"
   정도로 단순하다 (뚜껑돌리기 태스크가 들어와서 "토크 부호->회전 방향,
   회전수->수량"까지 표현해야 하는 시점이 되면, 그때는 조합이 다양해져서
   템플릿만으론 부자연스러워질 가능성이 높다 -- 그때 Claude API로 바꾸는
   게 맞다고 본다).

그래서 지금은: 문장 템플릿 여러 개 중 하나를 결정적으로 골라 다양성만
살짝 주고, 강도어(살짝/보통/세게)는 force_max의 **경험적 분포**(현재
episodes의 33/66 백분위수)로 정한다 -- 절대값을 하드코딩하지 않는 이유는
씬/게인 분포가 바뀌면 force_max 스케일 자체가 달라질 수 있어서다.

회전 방향/수량 라벨링(스펙에 언급된 "토크 부호로 회전 방향을, 회전수로
수량을")은 아직 구현하지 않는다 -- 뚜껑돌리기 같은 회전 태스크가 없어서
표현할 대상 자체가 없다.

사용 예:
    python pipeline/language_labeling.py --episodes-dir ./data/episodes
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pipeline.episode_io import load_episode, save_language

_TEMPLATES = [
    "오른손으로 peg를 구멍에 {intensity} 삽입하라.",
    "오른팔을 이용해 페그를 구멍 안으로 {intensity} 밀어 넣어라.",
    "오른손으로 핀을 구멍 위치에 맞춰 {intensity} 꽂아라.",
]

_INTENSITY_WORDS = {"gentle": "살짝", "normal": "적당한 힘으로", "firm": "힘있게"}


def _intensity_bucket(force_max: float, low_thresh: float, high_thresh: float) -> str:
    if force_max <= low_thresh:
        return "gentle"
    if force_max <= high_thresh:
        return "normal"
    return "firm"


def make_language(force_max: float, low_thresh: float, high_thresh: float, template_idx: int) -> str:
    intensity = _INTENSITY_WORDS[_intensity_bucket(force_max, low_thresh, high_thresh)]
    template = _TEMPLATES[template_idx % len(_TEMPLATES)]
    return template.format(intensity=intensity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=str, default="./data/episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.episodes_dir, "episode_*.npz")))
    if not paths:
        print(f"[language_labeling] {args.episodes_dir}에 episode_*.npz가 없음 -- 4단계를 먼저 실행하세요.")
        return

    force_maxes = np.array([load_episode(p)["force_max"] for p in paths])
    low_thresh = float(np.percentile(force_maxes, 33))
    high_thresh = float(np.percentile(force_maxes, 66))
    print(
        f"[language_labeling] {len(paths)}개 에피소드, force_max 범위 "
        f"[{force_maxes.min():.1f}, {force_maxes.max():.1f}]N, "
        f"강도 경계(33/66 백분위): {low_thresh:.1f}N / {high_thresh:.1f}N"
    )

    bucket_counts = {"gentle": 0, "normal": 0, "firm": 0}
    for i, path in enumerate(paths):
        episode = load_episode(path)
        force_max = episode["force_max"]
        language = make_language(force_max, low_thresh, high_thresh, template_idx=i)
        save_language(path, language)
        bucket_counts[_intensity_bucket(force_max, low_thresh, high_thresh)] += 1

    print(f"[language_labeling] 강도 분포: {bucket_counts}")
    print(f"[language_labeling] {len(paths)}개 에피소드에 language 필드 추가 완료")


if __name__ == "__main__":
    main()
