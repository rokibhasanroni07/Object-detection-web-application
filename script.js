(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const COLORS = ["#8b7cff","#53d6bd","#ff9b70","#62a8ff","#f06ea9","#e6c45f","#70d6ff","#a9df65","#ff7c72","#c29cff"];
  const HISTORY_KEY = "visionai_detection_history_v1";
  const state = { model:null, source:"", image:null, stream:null, running:false, paused:false, raf:0, busy:false, threshold:.5, tracks:[], nextTrackId:1, lastPredictions:[], history:loadHistory(), lastSavedKey:"", fpsFrames:0, fpsStart:performance.now(), fps:0, lastInference:0, toastTimer:0, currentView:"detector" };
  const canvas = $("canvas"), ctx = canvas.getContext("2d", { willReadFrequently:false });
  let toastTimer;
  function toast(message){ const el=$("toast"); el.textContent=message; el.classList.add("show"); clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.classList.remove("show"),3200); }
  function setStatus(text, ready=false, error=false){$("statusText").textContent=text;$("statusDot").classList.toggle("ready",ready);$("statusDot").style.background=error?"var(--danger)":"";}
  function loadHistory(){try{const v=JSON.parse(localStorage.getItem(HISTORY_KEY)||"[]");return Array.isArray(v)?v:[]}catch{return[]}}
  function persistHistory(){try{localStorage.setItem(HISTORY_KEY,JSON.stringify(state.history.slice(0,100)))}catch{toast("Browser storage is full; history may not be saved.");}}
  function escapeHTML(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
  function niceName(s){return String(s||"object").replace(/[-_]/g," ");}
  function colorFor(name){let n=0;for(const c of name)n=(n*31+c.charCodeAt(0))>>>0;return COLORS[n%COLORS.length]}
  async function init(){
    if(!window.tf || !window.cocoSsd){setStatus("Model library unavailable",false,true);$("modelState").textContent="Check internet connection";toast("Could not load TensorFlow.js. Check your internet connection and refresh.");return}
    try{
      setStatus("Loading AI model");$("modelState").textContent="Loading COCO-SSD…";
      await tf.ready();
      state.model=await cocoSsd.load({base:"lite_mobilenet_v2"});
      $("modelState").textContent="Ready · MobileNet V2";$("modelState").previousElementSibling?.classList.add("ready");
      setStatus("AI model ready",true);toast("AI model loaded. Upload an image or start your webcam.");renderAnalytics();
    }catch(err){console.error(err);setStatus("Model failed to load",false,true);$("modelState").textContent="Could not load model";toast("AI model failed to load. Check internet access and refresh the page.");}
  }
  function showSource(source){state.source=source;$("sourceBadge").textContent=source?source.toUpperCase():"NO SOURCE";$("canvasStage").classList.toggle("has-content",!!source);$("liveBadge").classList.toggle("hidden",source!=="webcam");$("downloadButton").disabled=!source||!canvas.width;}
  function resizeCanvas(w,h){const max=1600,scale=Math.min(1,max/Math.max(w,h));canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));}
  function fitBox(b,scaleX,scaleY){return [b[0]*scaleX,b[1]*scaleY,b[2]*scaleX,b[3]*scaleY]}
  function drawPredictions(predictions, sourceW, sourceH, tracking=true){
    const sx=canvas.width/sourceW, sy=canvas.height/sourceH;
    ctx.lineWidth=Math.max(2,canvas.width/450);ctx.font=`600 ${Math.max(12,Math.min(18,canvas.width/55))}px DM Sans, sans-serif`;ctx.textBaseline="bottom";
    predictions.forEach(p=>{
      const [x,y,w,h]=fitBox(p.bbox,sx,sy), color=colorFor(p.class);
      ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,canvas.width/450);ctx.strokeRect(x,y,w,h);
      const label=$("labelsToggle").checked?`${niceName(p.class)} ${(p.score*100).toFixed(0)}%${tracking&&p.trackId?` · #${p.trackId}`:""}`:`${niceName(p.class)}${tracking&&p.trackId?` · #${p.trackId}`:""}`;
      const tw=ctx.measureText(label).width, th=Math.max(21,canvas.width/60), lx=Math.max(0,Math.min(x,canvas.width-tw-12)), ly=Math.max(th, y);
      ctx.fillStyle=color;ctx.fillRect(lx,ly-th,tw+12,th);ctx.fillStyle="#08101d";ctx.fillText(label,lx+6,ly-4);
    });
  }
  function drawImageBase(img){resizeCanvas(img.naturalWidth||img.width,img.naturalHeight||img.height);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);}
  function updateMetrics(predictions,ms){
    $("objectMetric").textContent=predictions.length;$("classMetric").textContent=new Set(predictions.map(p=>p.class)).size;
    $("confidenceMetric").textContent=predictions.length?Math.round(predictions.reduce((s,p)=>s+p.score,0)/predictions.length*100)+"%":"—";
    $("inferenceMetric").textContent=Number.isFinite(ms)?Math.round(ms)+" ms":"—";$("resultCount").textContent=`${predictions.length} OBJECT${predictions.length===1?"":"S"}`;
    const list=$("objectList");
    if(!predictions.length){list.innerHTML='<div class="results-empty">No objects above the confidence threshold. Try lowering it.</div>';return}
    list.innerHTML=predictions.map(p=>`<div class="object-row"><span class="object-swatch" style="background:${colorFor(p.class)}"></span><div><div class="object-name">${escapeHTML(niceName(p.class))}${p.trackId?` <span style="color:#8c82d6">#${p.trackId}</span>`:""}</div><div class="object-meta">COCO-SSD prediction</div></div><span class="object-score">${Math.round(p.score*100)}%</span></div>`).join("");
  }
  function saveRun(source,predictions,ms){
    const key=source+":"+predictions.map(p=>p.class+Math.round(p.score*100)).sort().join(",");
    if(source==="webcam"&&key===state.lastSavedKey)return;
    state.lastSavedKey=key;
    const record={id:Date.now(),date:new Date().toISOString(),source,objects:predictions.length,avg:predictions.length?predictions.reduce((a,p)=>a+p.score,0)/predictions.length:0,classes:predictions.map(p=>({name:p.class,score:p.score,trackId:p.trackId||null}))};
    state.history.unshift(record);state.history=state.history.slice(0,100);persistHistory();renderAnalytics();
  }
  async function runImageDetection(img,fileName="uploaded image"){
    stopWebcam(false);state.image=img;showSource("image");$("canvasHint").textContent=fileName;
    try{
      setStatus("Detecting objects",true);drawImageBase(img);
      const start=performance.now();const raw=await state.model.detect(img);const ms=performance.now()-start;
      const predictions=raw.filter(p=>p.score>=state.threshold).map(p=>({...p}));state.lastPredictions=predictions;
      drawImageBase(img);drawPredictions(predictions,img.naturalWidth||img.width,img.naturalHeight||img.height,$("trackingToggle").checked);
      updateMetrics(predictions,ms);saveRun("image",predictions,ms);setStatus("Detection complete",true);$("fpsBadge").textContent="IMAGE";
    }catch(err){console.error(err);setStatus("Detection error",false,true);toast("Could not process this image. Try a different image file.");}
  }
  async function onUpload(file){
    if(!file)return;
    if(!file.type.startsWith("image/")){toast("Please choose a valid image file.");return}
    if(file.size>15*1024*1024){toast("This file is larger than 15 MB. Please choose a smaller image.");return}
    if(!state.model){toast("The AI model is still loading. Please wait.");return}
    const url=URL.createObjectURL(file),img=new Image();
    img.onload=async()=>{try{await runImageDetection(img,file.name)}finally{URL.revokeObjectURL(url)}};
    img.onerror=()=>{URL.revokeObjectURL(url);toast("This image could not be opened. Try JPG, PNG, or WEBP.");};
    img.src=url;
  }
  async function startWebcam(){
    if(!state.model){toast("Wait for the AI model to finish loading.");return}
    if(!navigator.mediaDevices?.getUserMedia){toast("Camera access is not supported here. Use HTTPS or localhost.");return}
    stopWebcam(false);
    try{
      state.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});
      const video=$("webcamVideo");video.srcObject=state.stream;await video.play();
      await new Promise(resolve=>{if(video.videoWidth)resolve();else video.addEventListener("loadedmetadata",resolve,{once:true})});
      state.running=true;state.paused=false;state.tracks=[];state.nextTrackId=1;state.lastSavedKey="";
      resizeCanvas(video.videoWidth,video.videoHeight);showSource("webcam");$("canvasHint").textContent="Live camera feed";$("webcamButton").disabled=true;$("pauseButton").disabled=false;$("stopButton").disabled=false;$("pauseButton").textContent="Ⅱ Pause";
      setStatus("Camera live",true);state.fpsFrames=0;state.fpsStart=performance.now();loopWebcam();
    }catch(err){console.error(err);stopWebcam(false);let msg=err.name==="NotAllowedError"?"Camera permission denied. Allow camera access in your browser.":err.name==="NotFoundError"?"No camera was found on this device.":"Could not start camera. Check permissions and try again.";toast(msg);setStatus("Camera unavailable",false,true)}
  }
  function trackPredictions(predictions){
    if(!$("trackingToggle").checked){state.tracks=[];return predictions}
    const used=new Set(),next=[];
    const tracked=predictions.map(p=>{
      const [x,y,w,h]=p.bbox,cx=x+w/2,cy=y+h/2;
      let best=null,bestDist=Infinity;
      for(const t of state.tracks){if(used.has(t.id)||t.class!==p.class)continue;const d=Math.hypot(cx-t.cx,cy-t.cy);const limit=Math.max(55,Math.hypot(w,h)*.8);if(d<limit&&d<bestDist){best=t;bestDist=d}}
      const id=best?best.id:state.nextTrackId++;if(best)used.add(id);
      next.push({id,class:p.class,cx,cy,bbox:p.bbox});return {...p,trackId:id};
    });
    state.tracks=next;return tracked;
  }
  async function loopWebcam(){
    if(!state.running||state.paused||state.busy)return;
    const video=$("webcamVideo");if(video.readyState<2){state.raf=requestAnimationFrame(loopWebcam);return}
    state.busy=true;
    try{
      const start=performance.now(),raw=await state.model.detect(video),ms=performance.now()-start;
      if(!state.running)return;
      const filtered=raw.filter(p=>p.score>=state.threshold);
      const predictions=trackPredictions(filtered);state.lastPredictions=predictions;state.lastInference=ms;
      ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(video,0,0,canvas.width,canvas.height);drawPredictions(predictions,video.videoWidth,video.videoHeight,$("trackingToggle").checked);
      updateMetrics(predictions,ms);
      state.fpsFrames++;const now=performance.now();if(now-state.fpsStart>=1000){state.fps=Math.round(state.fpsFrames*1000/(now-state.fpsStart));state.fpsFrames=0;state.fpsStart=now;$("fpsBadge").textContent=`${state.fps} FPS`}
      // Save one record per meaningful scene snapshot, not every frame.
      const signature=predictions.map(p=>p.class).sort().join(",");
      if(signature!==state.lastSceneSignature){state.lastSceneSignature=signature;saveRun("webcam",predictions,ms)}
    }catch(err){console.error(err);toast("Detection hit an error. Stop and restart the webcam if this continues.");stopWebcam();return}
    finally{state.busy=false}
    if(state.running&&!state.paused)state.raf=requestAnimationFrame(loopWebcam);
  }
  function pauseWebcam(){
    if(!state.running)return;
    state.paused=!state.paused;$("pauseButton").textContent=state.paused?"▶ Resume":"Ⅱ Pause";
    if(state.paused){cancelAnimationFrame(state.raf);setStatus("Camera paused",true)}else{setStatus("Camera live",true);loopWebcam()}
  }
  function stopWebcam(update=true){
    state.running=false;state.paused=false;state.busy=false;cancelAnimationFrame(state.raf);
    if(state.stream){state.stream.getTracks().forEach(t=>t.stop());state.stream=null}
    const video=$("webcamVideo");video.pause();video.srcObject=null;
    $("webcamButton").disabled=false;$("pauseButton").disabled=true;$("stopButton").disabled=true;$("pauseButton").textContent="Ⅱ Pause";
    if(state.source==="webcam"){state.source="";showSource("");ctx.clearRect(0,0,canvas.width,canvas.height);$("emptyState").style.display="";$("canvasStage").classList.remove("has-content");$("fpsBadge").textContent="— FPS";$("canvasHint").textContent="Supported: JPG, PNG, WEBP"}
    if(update&&state.model)setStatus("AI model ready",true);
  }
  function downloadResult(){
    if(!canvas.width||!canvas.height)return;
    canvas.toBlob(blob=>{if(!blob){toast("Could not export the image.");return}const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=`visionai-detection-${Date.now()}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}, "image/png");
  }
  function renderAnalytics(){
    const runs=state.history,all=runs.flatMap(r=>r.classes||[]),counts={};all.forEach(c=>counts[c.name]=(counts[c.name]||0)+1);
    $("historyRuns").textContent=runs.length;$("historyObjects").textContent=all.length;
    const top=Object.entries(counts).sort((a,b)=>b[1]-a[1])[0];$("topClass").textContent=top?niceName(top[0]):"—";
    $("historyConfidence").textContent=all.length?Math.round(all.reduce((s,c)=>s+c.score,0)/all.length*100)+"%":"—";
    const chart=$("classChart"),entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,8),max=entries[0]?.[1]||1;
    chart.innerHTML=entries.length?entries.map(([name,count])=>`<div class="bar-item"><div class="bar-label"><span>${escapeHTML(niceName(name))}</span><b>${count}</b></div><div class="bar-track"><div class="bar-fill" style="width:${count/max*100}%"></div></div></div>`).join(""):'<div class="results-empty">Run a detection to start building analytics.</div>';
    const history=$("historyList");history.innerHTML=runs.length?runs.slice(0,20).map(r=>`<div class="history-item"><div><div class="history-title">${escapeHTML(r.source==="webcam"?"Webcam session":"Image detection")}</div><div class="history-sub">${new Date(r.date).toLocaleString()} · ${(r.classes||[]).slice(0,3).map(c=>niceName(c.name)).join(", ")||"No objects detected"}</div></div><div class="history-number">${r.objects} obj.</div></div>`).join(""):'<div class="results-empty">No detection history yet.</div>';
  }
  function exportHistory(){
    if(!state.history.length){toast("There is no history to export yet.");return}
    const rows=[["Date","Source","Object count","Average confidence","Classes"],...state.history.map(r=>[r.date,r.source,r.objects,Math.round(r.avg*100)+"%",(r.classes||[]).map(c=>`${c.name} (${Math.round(c.score*100)}%)`).join("; ")])];
    const csv=rows.map(row=>row.map(v=>`"${String(v??"").replace(/"/g,'""')}"`).join(",")).join("\r\n");
    const blob=new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8;"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="visionai-detection-history.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function switchView(view){
    state.currentView=view;$("detectorView").classList.toggle("hidden",view!=="detector");$("analyticsView").classList.toggle("hidden",view!=="analytics");
    document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b.dataset.view===view));$("pageTitle").textContent=view==="detector"?"Object detector":"Analytics & history";if(view==="analytics")renderAnalytics();
  }
  $("imageUpload").addEventListener("change",e=>{onUpload(e.target.files?.[0]);e.target.value=""});
  $("uploadZone").addEventListener("dragover",e=>{e.preventDefault();$("uploadZone").classList.add("dragover")});
  $("uploadZone").addEventListener("dragleave",()=>$("uploadZone").classList.remove("dragover"));
  $("uploadZone").addEventListener("drop",e=>{e.preventDefault();$("uploadZone").classList.remove("dragover");onUpload(e.dataTransfer.files?.[0])});
  $("webcamButton").addEventListener("click",startWebcam);$("pauseButton").addEventListener("click",pauseWebcam);$("stopButton").addEventListener("click",()=>stopWebcam());
  $("downloadButton").addEventListener("click",downloadResult);$("threshold").addEventListener("input",e=>{state.threshold=Number(e.target.value)/100;$("thresholdValue").textContent=e.target.value+"%";if(state.source==="image"&&state.image)runImageDetection(state.image,$("canvasHint").textContent)});
  $("trackingToggle").addEventListener("change",()=>{state.tracks=[];state.nextTrackId=1});$("labelsToggle").addEventListener("change",()=>{if(state.source==="image"&&state.image)runImageDetection(state.image,$("canvasHint").textContent)});
  document.querySelectorAll(".nav-item").forEach(b=>b.addEventListener("click",()=>switchView(b.dataset.view)));
  $("themeButton").addEventListener("click",()=>{document.body.classList.toggle("light");localStorage.setItem("visionai_theme",document.body.classList.contains("light")?"light":"dark")});
  $("clearHistoryButton").addEventListener("click",()=>{if(!state.history.length){toast("History is already empty.");return}if(confirm("Clear all detection history stored in this browser?")){state.history=[];persistHistory();renderAnalytics();toast("Detection history cleared.")}});
  $("exportHistoryButton").addEventListener("click",exportHistory);
  try{if(localStorage.getItem("visionai_theme")==="light")document.body.classList.add("light")}catch{}
  window.addEventListener("beforeunload",()=>stopWebcam(false));
  init();
})();