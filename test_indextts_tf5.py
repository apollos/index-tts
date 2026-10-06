from pathlib import Path
import soundfile as sf
import torch
from indextts.infer_v2_5 import IndexTTS2

root = Path.home() / "local-model-workbench"
model_dir = root / "models/IndexTeam/IndexTTS-2.5"
reference = root / "artifacts/qwen3tts-test/my_ref.wav"
output = root / "artifacts/indextts25-test/indextts25_tf5_comfyui_native.wav"

for path in (model_dir / "config.yaml", reference):
    if not path.is_file():
        raise FileNotFoundError(path)
if not torch.cuda.is_available():
    raise RuntimeError("ROCm GPU 不可用")
output.parent.mkdir(parents=True, exist_ok=True)

tts = IndexTTS2(
    cfg_path=str(model_dir / "config.yaml"),
    model_dir=str(model_dir),
    device="cuda:0",
    use_bf16=False,
    use_cuda_kernel=False,
    use_deepspeed=False,
    use_qwen_emo=False,
)
result = tts.infer(
    spk_audio_prompt=str(reference),
    text="你好，这是 IndexTTS 在 Python 三点一二环境中的语音生成测试。",
    lang="ZH",
    output_path=None,
    verbose=True,
)
if result is None:
    raise RuntimeError("IndexTTS 没有返回音频")
sample_rate, audio = result
sf.write(str(output), audio, sample_rate, subtype="PCM_16")
print("输出文件:", output)
print("音频信息:", sf.info(str(output)))
