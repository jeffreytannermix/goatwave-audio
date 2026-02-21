import sys
from unittest.mock import MagicMock
for mod in ['numba','numba.core','numba.core.decorators','numba.typed','numba.np','numba.np.ufunc']:
    sys.modules[mod] = MagicMock()

import os
import subprocess
import numpy as np
import threading
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = Path.home() / 'stemsplitter' / 'uploads'
OUTPUT_FOLDER = Path.home() / 'separated'
MIX_FOLDER    = Path.home() / 'stemsplitter' / 'mixes'
AT_FOLDER     = Path.home() / 'stemsplitter' / 'autotuned'
for f in [UPLOAD_FOLDER, OUTPUT_FOLDER, MIX_FOLDER, AT_FOLDER]:
    f.mkdir(parents=True, exist_ok=True)

jobs = {}

def load_audio(path, sr=44100):
    import librosa
    y, rate = librosa.load(str(path), sr=sr, mono=True)
    return y, rate

def save_audio(y, sr, path):
    import soundfile as sf
    sf.write(str(path), y, sr)

def apply_highpass(y, sr, cutoff=80):
    from scipy.signal import butter, sosfilt
    if cutoff <= 0: return y
    sos = butter(4, cutoff / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, y)

def apply_eq(y, sr, low_db=0, high_db=0):
    from scipy.signal import butter, sosfilt
    result = y.copy()
    if abs(low_db) > 0.1:
        sos = butter(2, 300 / (sr / 2), btype='low', output='sos')
        result += sosfilt(sos, y) * (10 ** (low_db / 20) - 1)
    if abs(high_db) > 0.1:
        sos = butter(2, 5000 / (sr / 2), btype='high', output='sos')
        result += sosfilt(sos, y) * (10 ** (high_db / 20) - 1)
    return result

def apply_compression(y, threshold_db=-18, ratio=4.0):
    threshold = 10 ** (threshold_db / 20)
    out = y.copy()
    above = np.abs(y) > threshold
    out[above] = np.sign(y[above]) * (threshold + (np.abs(y[above]) - threshold) / ratio)
    return out

def apply_reverb(y, sr, mix=0.15):
    if mix <= 0: return y
    wet = np.zeros_like(y)
    for delay_s, decay in [(0.03, 0.4), (0.05, 0.3), (0.08, 0.2), (0.13, 0.1)]:
        d = int(sr * delay_s)
        tmp = np.zeros_like(y)
        tmp[d:] = y[:-d] * decay
        wet += tmp
    return y * (1 - mix) + wet * mix

def apply_delay(y, sr, time=0.25, mix=0.1):
    if mix <= 0: return y
    d = int(sr * time)
    delayed = np.zeros_like(y)
    if d < len(y):
        delayed[d:] = y[:-d] * 0.6
    return y + delayed * mix

def master_limiter(y, ceiling=0.95):
    peak = np.max(np.abs(y))
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y

def analyze_audio(path):
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(np.atleast_1d(tempo)[0])))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        notes  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        major  = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor  = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        best_score, best_key = -2, 'C Major'
        for i in range(12):
            rot = np.roll(chroma, -i)
            for profile, label in [(major, 'Major'), (minor, 'Minor')]:
                score = float(np.corrcoef(rot, profile)[0, 1])
                if score > best_score:
                    best_score, best_key = score, f'{notes[i]} {label}'
        mins, secs = int(duration // 60), int(duration % 60)
        return {'key': best_key, 'bpm': bpm, 'duration': f'{mins}:{secs:02d}'}
    except Exception as e:
        return {'key': '?', 'bpm': '?', 'duration': '?', 'error': str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    job_id = str(uuid.uuid4())
    path = UPLOAD_FOLDER / f'{job_id}{Path(f.filename).suffix}'
    f.save(str(path))
    return jsonify(analyze_audio(path))

def run_demucs(job_id, file_path, stems):
    try:
        jobs[job_id]['status'] = 'analyzing'
        jobs[job_id]['analysis'] = analyze_audio(file_path)
        jobs[job_id]['status'] = 'processing'
        cmd = [sys.executable, '-m', 'demucs']
        if stems and stems != 'all':
            cmd += ['--two-stems', stems]
        cmd.append(str(file_path))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            song_name = Path(file_path).stem
            out_dir = OUTPUT_FOLDER / 'htdemucs' / song_name
            stem_files = list(out_dir.glob('*.wav'))
            jobs[job_id].update({'status': 'done', 'output_dir': str(out_dir), 'files': [f.name for f in stem_files]})
        else:
            jobs[job_id].update({'status': 'error', 'error': result.stderr[-500:] or 'Demucs failed'})
    except Exception as e:
        jobs[job_id].update({'status': 'error', 'error': str(e)})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    stems = request.form.get('stems', 'all')
    job_id = str(uuid.uuid4())
    path = UPLOAD_FOLDER / f'{job_id}{Path(f.filename).suffix}'
    f.save(str(path))
    jobs[job_id] = {'status': 'queued', 'filename': f.filename}
    t = threading.Thread(target=run_demucs, args=(job_id, path, stems))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    return jsonify(job)

@app.route('/download/<job_id>/<filename>')
def download(job_id, filename):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done': return jsonify({'error': 'Not ready'}), 404
    return send_from_directory(job['output_dir'], filename, as_attachment=True)

KEY_SCALES = {
    'C':[0,2,4,5,7,9,11],'C#':[1,3,5,6,8,10,0],'D':[2,4,6,7,9,11,1],
    'D#':[3,5,7,8,10,0,2],'E':[4,6,8,9,11,1,3],'F':[5,7,9,10,0,2,4],
    'F#':[6,8,10,11,1,3,5],'G':[7,9,11,0,2,4,6],'G#':[8,10,0,1,3,5,7],
    'A':[9,11,1,2,4,6,8],'A#':[10,0,2,3,5,7,9],'B':[11,1,3,4,6,8,10],
}

def detect_pitch_fft(chunk, sr):
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed, n=8192))
    freqs = np.fft.rfftfreq(8192, 1 / sr)
    mask = (freqs >= 80) & (freqs <= 1200)
    if not mask.any(): return None
    freq = freqs[mask][np.argmax(spectrum[mask])]
    return float(freq) if freq > 0 else None

def autotune_audio(y, sr, key='C', correction=0.8):
    try:
        import librosa
        scale = KEY_SCALES.get(key, KEY_SCALES['C'])
        hop, frame_len = 1024, 4096
        output = y.copy()
        for start in range(0, len(y) - frame_len, hop):
            chunk = y[start:start + frame_len]
            freq = detect_pitch_fft(chunk, sr)
            if freq is None: continue
            midi = librosa.hz_to_midi(freq)
            semitone = midi % 12
            nearest = min(scale, key=lambda n: min(abs(n - semitone), 12 - abs(n - semitone)))
            diff = nearest - semitone
            if abs(diff) > 6: diff = diff - 12 if diff > 0 else diff + 12
            shift = diff * correction
            if abs(shift) < 0.05: continue
            try:
                shifted = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=shift)
                end = min(start + frame_len, len(output))
                output[start:end] = shifted[:end - start]
            except Exception:
                pass
        return output
    except Exception:
        return y

def run_autotune(job_id, path, key, correction):
    try:
        y, sr = load_audio(path)
        out = AT_FOLDER / f'{job_id}_autotuned.wav'
        save_audio(autotune_audio(y, sr, key, correction), sr, out)
        jobs[job_id] = {'status': 'done', 'job_id': job_id, 'file': out.name}
    except Exception as e:
        jobs[job_id] = {'status': 'error', 'error': str(e)}

@app.route('/autotune', methods=['POST'])
def autotune():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    key = request.form.get('key', 'C')
    correction = float(request.form.get('correction', 0.8))
    job_id = str(uuid.uuid4())
    path = UPLOAD_FOLDER / f'{job_id}_at{Path(f.filename).suffix}'
    f.save(str(path))
    jobs[job_id] = {'status': 'processing'}
    t = threading.Thread(target=run_autotune, args=(job_id, path, key, correction))
    t.daemon = True; t.start()
    return jsonify({'job_id': job_id})

@app.route('/autotune_download/<job_id>/<filename>')
def autotune_download(job_id, filename):
    return send_from_directory(str(AT_FOLDER), filename, as_attachment=True)

def run_mix(job_id, vox_path, inst_path, params, at_enabled, at_key, at_correction):
    try:
        y_vox, sr = load_audio(vox_path)
        y_inst, _ = load_audio(inst_path, sr=sr)
        if at_enabled:
            y_vox = autotune_audio(y_vox, sr, at_key, at_correction)
        y_vox = apply_highpass(y_vox, sr, float(params.get('highpass', 80)))
        y_vox = apply_compression(y_vox, float(params.get('comp_threshold', -18)))
        y_vox = apply_eq(y_vox, sr, float(params.get('low', 0)), float(params.get('high', 2)))
        y_vox = apply_reverb(y_vox, sr, float(params.get('reverb_mix', 0.15)))
        y_vox = apply_delay(y_vox, sr, float(params.get('delay_time', 0.25)), float(params.get('delay_mix', 0.1)))
        y_vox *= float(params.get('vocal_volume', 1.0))
        ml = max(len(y_vox), len(y_inst))
        y_vox = np.pad(y_vox, (0, ml - len(y_vox)))
        y_inst = np.pad(y_inst, (0, ml - len(y_inst)))
        mix = master_limiter(y_vox + y_inst * float(params.get('instrumental_volume', 0.9)))
        out = MIX_FOLDER / f'{job_id}_mix.wav'
        save_audio(mix, sr, out)
        jobs[job_id] = {'status': 'done', 'file': out.name}
    except Exception as e:
        jobs[job_id] = {'status': 'error', 'error': str(e)}

@app.route('/mix', methods=['POST'])
def mix():
    if 'vocals' not in request.files or 'instrumental' not in request.files:
        return jsonify({'error': 'Need both files'}), 400
    job_id = str(uuid.uuid4())
    vox = request.files['vocals']
    inst = request.files['instrumental']
    vox_path  = UPLOAD_FOLDER / f'{job_id}_vox{Path(vox.filename).suffix}'
    inst_path = UPLOAD_FOLDER / f'{job_id}_inst{Path(inst.filename).suffix}'
    vox.save(str(vox_path)); inst.save(str(inst_path))
    params = dict(request.form)
    at_enabled    = request.form.get('at_enabled', 'false') == 'true'
    at_key        = request.form.get('at_key', 'C')
    at_correction = float(request.form.get('at_correction', 0.8))
    jobs[job_id]  = {'status': 'processing'}
    t = threading.Thread(target=run_mix, args=(job_id, vox_path, inst_path, params, at_enabled, at_key, at_correction))
    t.daemon = True; t.start()
    return jsonify({'job_id': job_id})

@app.route('/mix_status/<job_id>')
def mix_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    return jsonify(job)

@app.route('/mix_download/<job_id>/<filename>')
def mix_download(job_id, filename):
    return send_from_directory(str(MIX_FOLDER), filename, as_attachment=True)

def run_multitrack(job_id, track_paths, volumes, vocal_params, at_enabled, at_key):
    try:
        mixed, sr = None, 44100
        for name, path in track_paths.items():
            y, sr = load_audio(path, sr=sr)
            vol = float(volumes.get(name, 1.0))
            if name == 'vocals':
                if at_enabled:
                    y = autotune_audio(y, sr, at_key, float(vocal_params.get('at_correction', 0.8)))
                y = apply_highpass(y, sr, float(vocal_params.get('highpass', 80)))
                y = apply_compression(y, float(vocal_params.get('comp', -18)))
                y = apply_reverb(y, sr, float(vocal_params.get('reverb_mix', 0.15)))
                y = apply_delay(y, sr, 0.25, float(vocal_params.get('delay_mix', 0.1)))
            y *= vol
            if mixed is None:
                mixed = y
            else:
                ml = max(len(mixed), len(y))
                mixed = np.pad(mixed, (0, ml - len(mixed))) + np.pad(y, (0, ml - len(y)))
        if mixed is None:
            jobs[job_id] = {'status': 'error', 'error': 'No tracks'}; return
        out = MIX_FOLDER / f'{job_id}_multitrack.wav'
        save_audio(master_limiter(mixed), sr, out)
        jobs[job_id] = {'status': 'done', 'file': out.name}
    except Exception as e:
        jobs[job_id] = {'status': 'error', 'error': str(e)}

@app.route('/multitrack', methods=['POST'])
def multitrack():
    track_names = ['vocals','drums','bass','guitar','keys','other']
    track_paths, job_id = {}, str(uuid.uuid4())
    for name in track_names:
        if name in request.files:
            f = request.files[name]
            p = UPLOAD_FOLDER / f'{job_id}_{name}{Path(f.filename).suffix}'
            f.save(str(p)); track_paths[name] = p
    if not track_paths: return jsonify({'error': 'No tracks'}), 400
    volumes = {n: request.form.get(f'{n}_volume', 1.0) for n in track_names}
    vocal_params = {
        'highpass': request.form.get('vocals_highpass', 80),
        'comp': request.form.get('vocals_comp', -18),
        'reverb_mix': request.form.get('vocals_reverb_mix', 0.15),
        'delay_mix': request.form.get('vocals_delay_mix', 0.1),
        'at_correction': request.form.get('at_correction', 0.8),
    }
    at_enabled = request.form.get('vocals_at_enabled', 'false') == 'true'
    at_key = request.form.get('vocals_at_key', 'C')
    jobs[job_id] = {'status': 'processing'}
    t = threading.Thread(target=run_multitrack, args=(job_id, track_paths, volumes, vocal_params, at_enabled, at_key))
    t.daemon = True; t.start()
    return jsonify({'job_id': job_id})

# ─────────────────────────────────────────────────────────
# MASTERING
# ─────────────────────────────────────────────────────────

MASTER_FOLDER = Path.home() / 'stemsplitter' / 'mastered'
MASTER_FOLDER.mkdir(parents=True, exist_ok=True)

def apply_mid_eq(y, sr, mid_db=0, center_hz=1000):
    from scipy.signal import butter, sosfilt
    if abs(mid_db) < 0.1:
        return y
    low_cut = max(20, center_hz / 1.4)
    high_cut = min(sr / 2 - 100, center_hz * 1.4)
    sos_bp = butter(2, [low_cut / (sr / 2), high_cut / (sr / 2)], btype='band', output='sos')
    band = sosfilt(sos_bp, y)
    return y + band * (10 ** (mid_db / 20) - 1)

def apply_harmonic_exciter(y, amount=0.05):
    if amount <= 0:
        return y
    return y + np.tanh(y * (1 + amount * 5)) * amount * 0.3

def measure_lufs(y):
    rms = np.sqrt(np.mean(y ** 2))
    if rms == 0:
        return -70.0
    return 20 * np.log10(rms) - 0.691

def run_master(job_id, file_path, params):
    try:
        jobs[job_id] = {'status': 'processing'}
        y, sr = load_audio(file_path)
        # EQ
        y = apply_eq(y, sr, float(params.get('low_db', 0)), float(params.get('high_db', 1)))
        y = apply_mid_eq(y, sr, float(params.get('mid_db', 0)))
        # Compression
        y = apply_compression(y, float(params.get('comp_threshold', -12)), ratio=3.0)
        # Harmonic exciter
        y = apply_harmonic_exciter(y, float(params.get('exciter', 0.05)))
        # Loudness targeting
        target_lufs = float(params.get('target_lufs', -14))
        gain_db = target_lufs - measure_lufs(y)
        y = y * (10 ** (gain_db / 20))
        # True peak limiter
        ceiling_linear = 10 ** (float(params.get('ceiling', -0.3)) / 20)
        y = master_limiter(y, ceiling=ceiling_linear)
        out = MASTER_FOLDER / f'{job_id}_mastered.wav'
        save_audio(y, sr, out)
        jobs[job_id] = {'status': 'done', 'file': out.name}
    except Exception as e:
        jobs[job_id] = {'status': 'error', 'error': str(e)}

@app.route('/master', methods=['POST'])
def master():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    job_id = str(uuid.uuid4())
    path = UPLOAD_FOLDER / f'{job_id}_master{Path(f.filename).suffix}'
    f.save(str(path))
    jobs[job_id] = {'status': 'processing'}
    t = threading.Thread(target=run_master, args=(job_id, path, dict(request.form)))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/master_status/<job_id>')
def master_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    return jsonify(job)

@app.route('/master_download/<job_id>/<filename>')
def master_download(job_id, filename):
    return send_from_directory(str(MASTER_FOLDER), filename, as_attachment=True)


# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🐐 Goat Wave Audio is running!")
    print("👉 Open your browser: http://localhost:5000\n")
    app.run(debug=False, port=5000)


# ─────────────────────────────────────────────────────────
# MASTERING
# ─────────────────────────────────────────────────────────

MASTER_FOLDER = Path.home() / 'stemsplitter' / 'mastered'
MASTER_FOLDER.mkdir(parents=True, exist_ok=True)

def apply_stereo_width(y, width=1.0):
    """Widen or narrow stereo image. Works on mono too (returns mono)."""
    if y.ndim == 1:
        return y  # mono — skip
    mid  = (y[0] + y[1]) * 0.5
    side = (y[0] - y[1]) * 0.5 * width
    return np.array([mid + side, mid - side])

def apply_harmonic_exciter(y, amount=0.05):
    """Add subtle harmonic distortion for presence and air."""
    if amount <= 0: return y
    return y + np.tanh(y * 3) * amount

def calculate_lufs(y, sr):
    """Approximate integrated loudness (simplified K-weighting)."""
    from scipy.signal import butter, sosfilt
    # High-shelf pre-filter
    sos = butter(2, 1500 / (sr / 2), btype='high', output='sos')
    filtered = sosfilt(sos, y)
    mean_sq = np.mean(filtered ** 2)
    if mean_sq < 1e-10: return -70.0
    return float(10 * np.log10(mean_sq) - 0.691)

def run_master(job_id, file_path, params):
    try:
        jobs[job_id] = {'status': 'processing'}
        import librosa, soundfile as sf

        y, sr = librosa.load(str(file_path), sr=None, mono=False)
        # If stereo, keep stereo for width processing
        if y.ndim == 1:
            y_mono = y
        else:
            y_mono = librosa.to_mono(y)

        # Work on mono for all processing, apply width at end
        y_work = y_mono.copy()

        # 1. Multiband EQ
        low_db  = float(params.get('low_db', 0))
        mid_db  = float(params.get('mid_db', 0))
        high_db = float(params.get('high_db', 1))
        y_work = apply_eq(y_work, sr, low_db, high_db)
        # Mid band
        if abs(mid_db) > 0.1:
            from scipy.signal import butter, sosfilt
            sos_lo = butter(2, 300  / (sr / 2), btype='high', output='sos')
            sos_hi = butter(2, 5000 / (sr / 2), btype='low',  output='sos')
            mid_band = sosfilt(sos_hi, sosfilt(sos_lo, y_work))
            y_work += mid_band * (10 ** (mid_db / 20) - 1)

        # 2. Compression
        comp_thresh = float(params.get('comp_threshold', -12))
        y_work = apply_compression(y_work, comp_thresh, ratio=3.0)

        # 3. Harmonic exciter
        exciter = float(params.get('exciter', 0.05))
        y_work = apply_harmonic_exciter(y_work, exciter)

        # 4. Loudness normalization to target LUFS
        target_lufs = float(params.get('target_lufs', -14))
        current_lufs = calculate_lufs(y_work, sr)
        gain_db = target_lufs - current_lufs
        gain_db = max(-20, min(gain_db, 20))  # safety clamp
        y_work *= 10 ** (gain_db / 20)

        # 5. True peak limiter
        ceiling_db = float(params.get('ceiling', -0.3))
        ceiling_linear = 10 ** (ceiling_db / 20)
        y_work = master_limiter(y_work, ceiling=ceiling_linear)

        # 6. Stereo width (duplicate to stereo if mono)
        width = float(params.get('width', 1.0))
        y_stereo = np.array([y_work, y_work])  # mono → stereo
        y_stereo = apply_stereo_width(y_stereo, width)

        out_file = MASTER_FOLDER / f'{job_id}_mastered.wav'
        sf.write(str(out_file), y_stereo.T, sr)

        jobs[job_id] = {'status': 'done', 'file': out_file.name}
    except Exception as e:
        jobs[job_id] = {'status': 'error', 'error': str(e)}


@app.route('/master', methods=['POST'])
def master():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    job_id = str(uuid.uuid4())
    path = UPLOAD_FOLDER / f'{job_id}_master{Path(f.filename).suffix}'
    f.save(str(path))
    params = dict(request.form)
    jobs[job_id] = {'status': 'processing'}
    t = threading.Thread(target=run_master, args=(job_id, path, params))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/master_status/<job_id>')
def master_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    return jsonify(job)


@app.route('/master_download/<job_id>/<filename>')
def master_download(job_id, filename):
    return send_from_directory(str(MASTER_FOLDER), filename, as_attachment=True)
