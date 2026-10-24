Core idea:

Mimi streaming codec: 24 kHz waveform in, 12.5 frames/s discrete codes out.
Moshi full-duplex: two aligned audio streams (left / channel 0 = Moshi/agent, right / channel 1 = user), plus timestamped text.

Telephony and many open voice sets are 8 kHz or 16 kHz(Preferred). Resample them to 24 kHz so Mimi can encode them. Telugu is the proof of concept for sub-300 ms full-duplex


Plan of action:

•  Set up Moshi CPU environment — done

•  Test Mimi codec encoding/decoding on Telugu samples — done

•  Baseline environment running; validate Mimi reproduces Telugu phonemes clearly at 24 kHz — done

•  Gather 2–5 hours of dual-channel Telugu dialogue (separate agent/user tracks merged to stereo) — partially done (smoke MVP: 40 train/10 eval clips; notebook now builds turn-taking stereo + Roman Telugu text; scale to 2–5 hours still needed)

•  Resample to 24 kHz. Build timestamped JSONL in the moshi-finetune annotator format. Use AI4Bharat IndicXlit (`ai4bharat-transliteration`) for Telugu → Roman Telugu; WhisperX forced align (`language_code=te`) for real word times (even spacing only as fallback) — partially done (romanization + WhisperX wiring in `moshi_telugu.ipynb`; re-run Step 2 on Colab GPU to regenerate `stereo_mvp_roman_aligned`)

•  Rent a single NVIDIA GPU instance and install moshi-finetune there . MIG is overkill for baseline — done (verified on RunPod A100)

•  Launch LoRA fine-tuning with kyutai-labs/moshi-finetune (approx 10-14 hrs training). It encodes both stereo streams into Mimi tokens. Suggested: rank 64–128, batch size 8–16, gradient_checkpointing=True for better GPU efficiency — partially done (200-step smoke run completed to validate pipeline; full 10–14h training remaining)

•  Load adapters into Moshi’s PyTorch to finetune moshi instead of full training there by making base model frozen .  
python -m moshi.server \
  --lora-weight /path/to/lora.safetensors \
  --config-path /path/to/config.json. Merge the base model with telugu adapter if results look promising. — partially done (tested serving adapter via moshi.server & Gradio tunnel; weights merge pending full training)

•  Test with a live microphone with moshi ui. Record screen capture for demo. — partially done (verified live microphone input & Gradio tunnel connectivity; functional Telugu interaction pending proper data & training)

•  Continuous pre-training / more SFT on several thousand hours to handle edge cases. That needs more data and GPU-hours (multi-GPU or a larger instance) - not done

•  Further SFT from that checkpoint toward a stronger Telugu 7B. Stay on BF16 for this phase; FP8 is an optional later efficiency choice. Moshi is already multi-stream (text + audio codebooks) - not done

•  Scale to other Indic languages.  - not done




