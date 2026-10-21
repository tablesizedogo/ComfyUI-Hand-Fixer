# End-to-end test of the ComfyUI Hand Fixer node pack: headless ComfyUI + DreamShaper 8 inpainting + HADM-L.
import glob, json, os, shutil, subprocess, sys, tarfile, time, urllib.request
T0 = time.time(); timings = {}
OUT = '/kaggle/working/out'; os.makedirs(OUT, exist_ok=True)
def sh(cmd):
    t = time.time(); print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode: raise SystemExit(f'failed: {cmd}')
    return round(time.time() - t, 1)
inp = os.path.dirname(glob.glob('/kaggle/input/**/images/31632521.jpg', recursive=True)[0]).rsplit('/images', 1)[0]
print('input', inp, os.listdir(inp), flush=True)
W = '/kaggle/temp'; os.makedirs(W, exist_ok=True)
pack_src = glob.glob(inp + '/**/custom_nodes/comfyui-hand-fixer', recursive=True)
if not pack_src:
    with tarfile.open(glob.glob(inp + '/**/comfyui_hand_fixer.tgz', recursive=True)[0]) as t: t.extractall(W)
    pack_src = [W + '/comfyui_hand_fixer/custom_nodes/comfyui-hand-fixer']
pack_root = pack_src[0].rsplit('/custom_nodes', 1)[0]
C = W + '/ComfyUI'
timings['comfy_clone'] = sh(f'git clone -q https://github.com/comfyanonymous/ComfyUI.git {C} && git -C {C} checkout -q faa7e494f3a0c9e45e1c3db8d20ee48fd5fe1b52')
sh(f"grep -v -E '^(torch|torchvision|torchaudio)$' {C}/requirements.txt > {W}/req.txt")
timings['pip'] = sh(f'{sys.executable} -m pip install -q -r {W}/req.txt -r {pack_src[0]}/requirements.txt')
shutil.copytree(pack_src[0], C + '/custom_nodes/comfyui-hand-fixer')
sh(f'{sys.executable} -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"; pip list 2>/dev/null | grep -i -E "^(fairscale|xformers|setuptools|ultralytics|open.clip) "; nvidia-smi --query-gpu=name,memory.total --format=csv')
timings['download_hand_fixer_models'] = sh(f'{sys.executable} {C}/custom_nodes/comfyui-hand-fixer/download_models.py --sha256')
CK = 'DreamShaper_8_INPAINTING.inpainting.safetensors'
timings['download_sd15_inpaint'] = sh(f'curl -sSL --retry 3 -o {C}/models/checkpoints/{CK} https://huggingface.co/Lykon/DreamShaper/resolve/main/{CK} && ls -l {C}/models/checkpoints && sha256sum {C}/models/checkpoints/{CK}')
log = open(OUT + '/comfyui.log', 'w')
srv = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--disable-auto-launch'], cwd=C, stdout=log, stderr=subprocess.STDOUT)
API = 'http://127.0.0.1:8188'
for _ in range(180):
    try: urllib.request.urlopen(API + '/object_info/HandFixerBestOfN', timeout=5); break
    except Exception: time.sleep(2)
else: raise SystemExit('ComfyUI did not start')
print('ComfyUI up', round(time.time() - T0), flush=True)
def run(wf, tag):
    t = time.time()
    r = urllib.request.Request(API + '/prompt', data=json.dumps({'prompt': wf}).encode(), headers={'Content-Type': 'application/json'})
    try: pid = json.loads(urllib.request.urlopen(r).read())['prompt_id']
    except urllib.error.HTTPError as e: print(e.read().decode()[:3000]); raise
    while True:
        h = json.loads(urllib.request.urlopen(f'{API}/history/{pid}').read())
        if pid in h and h[pid].get('status', {}).get('completed') is not None or (pid in h and h[pid].get('status', {}).get('status_str') == 'error'):
            break
        time.sleep(2)
    h = h[pid]; st = h.get('status', {})
    print(tag, st.get('status_str'), round(time.time() - t, 1), 's', flush=True)
    if st.get('status_str') != 'success': print(json.dumps(st)[:4000], flush=True)
    return h, round(time.time() - t, 1)
wf0 = json.load(open(pack_root + '/workflows/hand_fix_sd15_inpaint_api.json'))
results = {}
for i, p in enumerate(sorted(glob.glob(inp + '/**/images/*.jpg', recursive=True))):
    name = os.path.basename(p); shutil.copy(p, f'{C}/input/{name}')
    wf = json.loads(json.dumps(wf0)); wf['2']['inputs']['image'] = name
    for k, pre in [('13', 'best'), ('14', 'preview'), ('15', 'candidates')]:
        wf[k]['inputs']['filename_prefix'] = f'hand_fixer/{name[:-4]}_{pre}'
    h, dt = run(wf, name)
    outs = h.get('outputs', {})
    results[name] = dict(status=h.get('status', {}).get('status_str'), seconds=dt,
                         scores=outs.get('4', {}).get('text'), best_of_n=outs.get('12', {}).get('text'))
    print(json.dumps(results[name])[:2000], flush=True)
timings['hadm_workflow_total'] = round(sum(r['seconds'] for r in results.values()), 1)
# light fallback backend, scorer only
wfc = {k: wf0[k] for k in ['2', '3', '4', '14']}
wfc['3'] = json.loads(json.dumps(wf0['3'])); wfc['3']['inputs']['backend'] = 'clip_l14_student (weaker, light)'
clip = {}
for p in sorted(glob.glob(f'{C}/input/*.jpg')):
    name = os.path.basename(p); w = json.loads(json.dumps(wfc)); w['2']['inputs']['image'] = name
    w['14'] = dict(class_type='SaveImage', inputs=dict(images=['4', 1], filename_prefix=f'hand_fixer/{name[:-4]}_clip_preview'))
    h, dt = run(w, 'clip ' + name); clip[name] = dict(status=h.get('status', {}).get('status_str'), seconds=dt, scores=h.get('outputs', {}).get('4', {}).get('text'))
timings['clip_workflow_total'] = round(sum(r['seconds'] for r in clip.values()), 1)
srv.terminate()
shutil.copytree(f'{C}/output/hand_fixer', OUT + '/images', dirs_exist_ok=True)
timings['total'] = round(time.time() - T0, 1)
json.dump(dict(hadm=results, clip=clip, timings=timings), open(OUT + '/report.json', 'w'), indent=1)
print(json.dumps(timings), flush=True)
