import sys
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders

MIMI_REPO = "kyutai/moshika-pytorch-bf16"


def test_mimi_codec(input_wav_path: str, output_wav_path: str = "telugu_decoded.wav"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Mimi on device: {device}...")

    # 1. Load pretrained Mimi weights bundled with Moshi
    mimi_weight = hf_hub_download(MIMI_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()

    # 2. Load Telugu audio
    print(f"Loading input audio from: {input_wav_path}")
    # torchaudio.load needs torchcodec on this Windows build; soundfile does not.
    audio, sr = sf.read(input_wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T.copy())  # [C, T]

    # 3. Resample to Mimi's native 24 kHz and convert to mono
    if sr != mimi.sample_rate:
        print(f"Resampling from {sr} Hz to {mimi.sample_rate} Hz...")
        wav = torchaudio.functional.resample(wav, sr, mimi.sample_rate)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Add batch dimension: [C, T] -> [1, C, T]
    wav = wav.unsqueeze(0).to(device)

    # 4. Run Encode -> Decode
    print("Encoding audio into Mimi discrete RVQ codes...")
    with torch.no_grad():
        # codes shape: [B, num_codebooks, time_steps]
        codes = mimi.encode(wav)
        print(f"Codebook shape: {tuple(codes.shape)} (Frame rate ~ 12.5 Hz)")

        print("Decoding codes back to audio waveform...")
        reconstructed_wav = mimi.decode(codes)

    # 5. Save output waveform
    out_audio = reconstructed_wav.squeeze(0).cpu().numpy()
    if out_audio.ndim == 2:
        out_audio = out_audio.T
    sf.write(output_wav_path, out_audio, mimi.sample_rate)
    print(f"\nSuccess! Reconstructed audio saved to: {output_wav_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_telugu_codec.py <path_to_telugu_audio.wav>")
    else:
        test_mimi_codec(sys.argv[1])
