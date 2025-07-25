// === Heart Rate Analyzer Front-end ===
// Restored complete UI logic (preview → crop → analyze → results)
class HeartRateUI {
    constructor() {
        this.cache();
        this.bind();
        // runtime state
        this.previewImage = null;
        this.crop = null;
        this.isSelecting = false;
        this.startX = 0;
        this.startY = 0;
        this.pollTimer = null;
    }

    // ---------- helpers ----------
    qs(id) { return document.getElementById(id); }
    show(el) { el.style.display = ''; }
    hide(el) { el.style.display = 'none'; }

    cache() {
        // inputs
        this.urlInput = this.qs('url-input');
        this.analyzeBtn = this.qs('analyze-btn');
        this.customTS = this.qs('custom-timestamps');
        this.tsControls = this.qs('timestamp-controls');
        this.startTime = this.qs('start-time');
        this.endTime = this.qs('end-time');
        this.enableTracking = this.qs('enable-tracking');

        // crop / preview
        this.resetCropBtn = this.qs('reset-crop-btn');
        this.previewSection = this.qs('preview-section');
        this.canvas = this.qs('preview-canvas');
        this.overlay = this.qs('crop-overlay');
        this.selection = this.qs('crop-selection');
        this.confirmCropBtn = this.qs('confirm-crop-btn');
        this.cancelCropBtn = this.qs('cancel-crop-btn');

        // progress
        this.progressSection = this.qs('progress-section');
        this.progressFill = this.qs('progress-fill');
        this.progressMsg = this.qs('progress-message');
        this.steps = {
            download: this.qs('step-download'),
            extract: this.qs('step-extract'),
            process: this.qs('step-process'),
            analyze: this.qs('step-analyze'),
            complete: this.qs('step-complete')
        };

        // results
        this.resultsSection = this.qs('results-section');
        this.bpmDisplay = this.qs('bpm-display');
        this.peaksCount = this.qs('peaks-count');
        this.analysisRange = this.qs('analysis-range');
        this.cropRegion = this.qs('crop-region');
        this.plotImage = this.qs('plot-image');
        this.overlayVideo = this.qs('overlay-video');
        this.magnifiedVideo = this.qs('magnified-video');
    }

    bind() {
        // validate URL input
        this.urlInput.addEventListener('input', () => this.validateURL());

        // main analyze / preview btn
        this.analyzeBtn.addEventListener('click', e => { e.preventDefault(); this.loadPreview(); });

        // check-box toggle for timestamps
        this.customTS.addEventListener('change', () => {
            this.tsControls.style.display = this.customTS.checked ? 'flex' : 'none';
        });

        // crop UI (selection active immediately after preview)
        this.resetCropBtn.addEventListener('click', () => this.resetCrop());
        this.confirmCropBtn.addEventListener('click', () => this.startAnalysis());
        this.cancelCropBtn.addEventListener('click', () => this.exitPreview());

        // canvas mouse events
        this.overlay.addEventListener('mousedown', e => this.beginSelect(e));
        this.overlay.addEventListener('mousemove', e => this.onSelect(e));
        this.overlay.addEventListener('mouseup', () => this.endSelect());
        this.overlay.addEventListener('mouseleave', () => this.endSelect());

        // pretty format time inputs on blur
        ;[this.startTime,this.endTime].forEach(inp=>{
            inp.addEventListener('blur',()=>{
                const secs=this.parseTime(inp.value);
                if(secs!=null){inp.value=this.formatSeconds(secs);}else inp.value='';
            });
        });
    }

    validateURL() {
        const url = this.urlInput.value.trim();
        const valid = /youtu\.be|youtube\.com/.test(url);
        this.analyzeBtn.disabled = !valid;
        this.analyzeBtn.querySelector('.btn-text').textContent = valid ? 'Load Preview' : 'Enter YouTube URL';
    }

    // ---------- Preview ----------
    async loadPreview() {
        const url = this.urlInput.value.trim();
        if (!url) return;
        // UI state
        this.disableInputs();
        this.progressMsg.textContent = 'Fetching preview frame…';
        this.show(this.progressSection);
        this.progressFill.style.width = '10%';
        this.setStep('download');

        const payload = { url };
        if (this.customTS.checked) {
            payload.timestamps = {
                start: this.parseTime(this.startTime.value),
                end: this.parseTime(this.endTime.value)
            };
        }
        try {
            const res = await fetch('/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            if (!res.ok) throw new Error('Preview failed');
            const data = await res.json();
            this.previewImage = data.preview_frame;
            this.showPreviewImage();
        } catch (err) {
            alert(err.message);
            this.enableInputs();
        }
    }

    showPreviewImage() {
        // draw image to canvas
        const img = new Image();
        img.onload = () => {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            const ctx = this.canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            // default crop is full frame
            this.crop = { x: 0, y: 0, width: img.width, height: img.height };
            this.hide(this.progressSection);
            this.show(this.previewSection);
            // immediately allow region selection
            this.startCropMode();
            this.enableInputs();
        };
        img.src = `data:image/jpeg;base64,${this.previewImage}`;
    }

    // ---------- Crop selection ----------
    startCropMode() {
        if (!this.previewImage) return alert('Load a preview first');
        this.selection.style.display = 'none';
        this.isSelecting = false;
        this.overlay.style.cursor = 'crosshair';
    }

    beginSelect(e) {
        if (!this.overlay.style.cursor.includes('crosshair')) return;
        const rect = this.overlay.getBoundingClientRect();
        this.startX = e.clientX - rect.left;
        this.startY = e.clientY - rect.top;
        this.isSelecting = true;
        this.selection.style.left = `${this.startX}px`;
        this.selection.style.top = `${this.startY}px`;
        this.selection.style.width = '0px';
        this.selection.style.height = '0px';
        this.selection.style.display = 'block';
    }

    onSelect(e) {
        if (!this.isSelecting) return;
        const rect = this.overlay.getBoundingClientRect();
        const currX = e.clientX - rect.left;
        const currY = e.clientY - rect.top;
        const left = Math.min(this.startX, currX);
        const top = Math.min(this.startY, currY);
        const width = Math.abs(currX - this.startX);
        const height = Math.abs(currY - this.startY);
        Object.assign(this.selection.style, { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });
    }

    endSelect() {
        if (!this.isSelecting) return;
        this.isSelecting = false;
        // convert to video coords
        const canvasRect = this.canvas.getBoundingClientRect();
        const overlayRect = this.overlay.getBoundingClientRect();
        const scaleX = this.canvas.width / canvasRect.width;
        const scaleY = this.canvas.height / canvasRect.height;
        const selRect = this.selection.getBoundingClientRect();
        const x = (selRect.left - overlayRect.left) * scaleX;
        const y = (selRect.top - overlayRect.top) * scaleY;
        const w = selRect.width * scaleX;
        const h = selRect.height * scaleY;
        if (w < 10 || h < 10) {
            // too small – ignore
            this.selection.style.display = 'none';
            alert('Selection too small – using full frame');
            this.crop = { x: 0, y: 0, width: this.canvas.width, height: this.canvas.height };
            return;
        }
        this.crop = { x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h) };
        this.resetCropBtn.style.display = '';
    }

    resetCrop() {
        this.selection.style.display = 'none';
        this.crop = { x: 0, y: 0, width: this.canvas.width, height: this.canvas.height };
        this.resetCropBtn.style.display = 'none';
    }

    exitPreview() {
        this.hide(this.previewSection);
    }

    // ---------- Analysis ----------
    async startAnalysis() {
        const url = this.urlInput.value.trim();
        const payload = { url, crop: this.crop, enable_tracking: this.enableTracking.checked };
        if (this.customTS.checked) {
            payload.timestamps = { start: this.parseTime(this.startTime.value), end: this.parseTime(this.endTime.value) };
        }
        this.hide(this.previewSection);
        this.resetSteps();
        this.progressFill.style.width = '0%';
        this.show(this.progressSection);
        this.setStep('download');
        this.disableInputs();
        await fetch('/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        this.pollTimer = setInterval(() => this.pollProgress(), 1200);
    }

    async pollProgress() {
        const res = await fetch('/progress');
        const data = await res.json();
        if (data.status === 'completed') {
            clearInterval(this.pollTimer);
            this.updateProgress(100, 'Complete');
            this.showResults(data.result);
            this.enableInputs();
        } else if (data.status === 'error') {
            clearInterval(this.pollTimer);
            alert(data.error);
            this.enableInputs();
        } else {
            this.updateProgress(data.progress || 0, data.message || '…');
        }
    }

    updateProgress(pct, msg) {
        this.progressFill.style.width = `${pct}%`;
        this.progressMsg.textContent = msg;
        if (pct < 30) this.setStep('download');
        else if (pct < 50) this.setStep('extract');
        else if (pct < 80) this.setStep('process');
        else if (pct < 95) this.setStep('analyze');
        else this.setStep('complete');
    }

    setStep(name) {
        Object.values(this.steps).forEach(s => s.classList.remove('active', 'completed'));
        if (name === 'complete') {
            Object.values(this.steps).forEach(s => s.classList.add('completed'));
        } else {
            let passed = true;
            for (const key of ['download','extract','process','analyze','complete']) {
                const el = this.steps[key];
                if (key === name) { el.classList.add('active'); passed = false; }
                else if (passed) el.classList.add('completed');
            }
        }
    }

    resetSteps() { Object.values(this.steps).forEach(s => s.classList.remove('active','completed')); }

    showResults(r) {
        this.hide(this.progressSection);
        this.bpmDisplay.textContent = `${r.bpm} BPM`;
        this.peaksCount.textContent = `${r.peaks_count} peaks`;
        // optionally show range inline beneath peaks
        this.peaksCount.insertAdjacentHTML('beforeend', `<br><span style="font-size:0.85rem;color:var(--text-muted)">${r.analysis_range}</span>`);
        this.plotImage.src = `data:image/png;base64,${r.plot_url}`;
        if (r.overlay_video) this.overlayVideo.src = `/videos/${r.overlay_video.split('/').pop()}`;
        if (r.magnified_video) this.magnifiedVideo.src = `/videos/${r.magnified_video.split('/').pop()}`;
        this.show(this.resultsSection);
    }

    disableInputs() { this.urlInput.disabled = this.analyzeBtn.disabled = true; }
    enableInputs() { this.urlInput.disabled = false; this.validateURL(); }

    parseTime(str) {
        const txt=str.trim();
        if(!txt) return null;
        if(!txt.includes(':')){
            // pure digits: treat len<=2 as seconds, else MMSS or HHMMSS
            if(/^[0-9]+$/.test(txt)){
                if(txt.length<=2){return parseInt(txt);} // seconds
                if(txt.length<=4){ // MMSS
                    const mins=parseInt(txt.slice(0,-2));
                    const secs=parseInt(txt.slice(-2));
                    return mins*60+secs;
                }
                if(txt.length<=6){ // HHMMSS
                    const hrs=parseInt(txt.slice(0,-4));
                    const mins=parseInt(txt.slice(-4,-2));
                    const secs=parseInt(txt.slice(-2));
                    return hrs*3600+mins*60+secs;
                }
            }
            return null;
        }
        const parts = txt.split(':').map(p=>p.trim());
        let total=0;
        if(parts.length===2){total=parseInt(parts[0])*60+parseFloat(parts[1]);}
        else if(parts.length===3){total=parseInt(parts[0])*3600+parseInt(parts[1])*60+parseFloat(parts[2]);}
        else return null;
        return isNaN(total)||total<0?null:total;
    }

    formatSeconds(total){
        total=Math.round(total);
        const m=Math.floor(total/60);const s=total%60;
        return `${m}:${s.toString().padStart(2,'0')}`;
    }
}

// --- util for video buttons -----
window.playVideo = type => {
    const vid = document.getElementById(`${type}-video`);
    if (vid) vid.paused ? vid.play() : vid.pause();
};
window.downloadVideo = type => {
    const vid = document.getElementById(`${type}-video`);
    if (vid && vid.src) {
        const a = document.createElement('a'); a.href = vid.src; a.download = vid.src.split('/').pop(); a.click();
    }
};

document.addEventListener('DOMContentLoaded', () => new HeartRateUI()); 