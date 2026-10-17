Core idea:

Mimi streaming codec: 24 kHz waveform in, 12.5 frames/s discrete codes out.
Moshi full-duplex: two aligned audio streams (left / channel 0 = Moshi/agent, right / channel 1 = user), plus timestamped text.

Telephony and many open voice sets are 8 kHz or 16 kHz(Preferred). Resample them to 24 kHz so Mimi can encode them. Telugu is the proof of concept for sub-300 ms full-duplex


Plan of action:

•  Set up Moshi CPU environment — done

•  Test Mimi codec encoding/decoding on Telugu samples — done

•  Baseline environment running; validate Mimi reproduces Telugu phonemes clearly at 24 kHz — done

•  Gather 2–5 hours of dual-channel Telugu dialogue (separate agent/user tracks merged to stereo) - done

•  Resample to 24 kHz. Build timestamped JSONL in the moshi-finetune annotator format. Use Indic-capable ASR and spot-check alignments before training (for sample training time stamps are fake so output will be gibberish) - done

•  Rent a single NVIDIA GPU instance and install moshi-finetune there . MIG is overkill for baseline - done

•  Launch LoRA fine-tuning with kyutai-labs/moshi-finetune (approx 10-14 hrs training). It encodes both stereo streams into Mimi tokens. Suggested: rank 64–128, batch size 8–16, gradient_checkpointing=True for better GPU efficiency.

•  Load adapters into Moshi’s PyTorch to finetune moshi instead of full training there by making base model frozen .  python -m moshi.server --base-model kyutai/moshiko-pytorch-bf16 --lora-path ./adapter.pt. Merge the base model with telugu adapter if results look promising. 

•  Test with a live microphone with moshi ui. Record screen capture for demo.

•  Continuous pre-training / more SFT on several thousand hours to handle edge cases. That needs more data and GPU-hours (multi-GPU or a larger instance)

•  Further SFT from that checkpoint toward a stronger Telugu 7B. Stay on BF16 for this phase; FP8 is an optional later efficiency choice. Moshi is already multi-stream (text + audio codebooks).

•  Scale to other Indic languages.




