# experiments/

반복 실험은 여기서 관리한다. 새 실험 계열마다 `experiments/<실험명>/` 폴더를 만들고,
그 실험의 실행 스크립트·로그·표를 그 안에 둔다. 재사용 가능한 핵심 코드(Models/, Utils/, FL/,
analysis_*.py, evaluate_*.py)는 지금처럼 저장소 루트에 그대로 둔다 — 상대 임포트가 루트 기준이라
스크립트를 옮기면 깨진다. `experiments/` 밑의 스크립트는 항상 **저장소 루트에서 실행**한다.

## evensplit_resnet8/

generalist를 균등분할하고 ResNet-8을 scratch로 학습시키는 실험 계열.
- `run_experiment.sh <n_generalists> <purity> [seed]` — 학습→로짓 생성→전체 지표 추출까지 한 번에 실행
- `logs_2026-09-29/` — 지금까지 완료한 5개 세팅(seed 90 purity 0.85/0.65/0.5, seed 91/92 purity 0.85)의 로그·표
- **세부 하이퍼파라미터·지표 설명은 `evensplit_resnet8/README.md` 참고**
