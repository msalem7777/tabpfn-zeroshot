/* ZeroShot: no framework, no browser-stored secrets, no external scripts. */
"use strict";
const $ = id => document.getElementById(id);
let state = null, csrf = "", editIndex = -1, polling = false, results = null;
let reviewSourceId = null;

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = String(text);
  if (className) el.className = className;
  return el;
}
function notify(message, error=false) {
  $("notice").hidden = false;
  $("notice").textContent = message;
  $("notice").className = error ? "error" : "";
  const dialog = document.querySelector("dialog[open]");
  if (dialog && error) {
    let alert = dialog.querySelector(".dialog-error");
    if (!alert) { alert = node("p", "", "dialog-error"); alert.setAttribute("role", "alert"); dialog.prepend(alert); }
    alert.textContent = message;
  }
}
async function api(path, body, method="POST") {
  const options = {method, headers:{}};
  if (method !== "GET") options.headers["X-CSRF-Token"] = csrf;
  if (body instanceof FormData) options.body = body;
  else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch("/api/" + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
function action(fn) {
  return async event => {
    event?.preventDefault();
    const b = event?.currentTarget;
    if (b?.tagName === "BUTTON") b.disabled = true;
    try { await fn(event); }
    catch (e) { notify(e.message, true); }
    finally { if (b?.tagName === "BUTTON") b.disabled = false; }
  };
}
function tab(name) {
  document.querySelectorAll(".tab").forEach(el => el.hidden = el.id !== "tab-" + name);
  document.querySelectorAll(".nav").forEach(el => el.classList.toggle("active", el.dataset.tab === name));
  if (name === "results" && state?.has_result) loadResults().catch(e => notify(e.message,true));
}
document.querySelectorAll("[data-tab]").forEach(el => el.onclick = () => tab(el.dataset.tab));
document.querySelectorAll("[data-close]").forEach(el => el.onclick = () => $(el.dataset.close).close());

function readTarget() {
  return {name: $("targetName").value.trim(), definition: $("targetDefinition").value.trim(),
    unit: $("targetUnit").value.trim(), population: $("targetPopulation").value.trim(),
    time_horizon: $("timeHorizon").value.trim(), family: $("targetFamily").value,
    positive_class: $("targetFamily").value === "logistic" ? $("positiveClass").value.trim() || null : null};
}
function fillTarget(t) {
  for (const [id,key] of [["targetName","name"],["targetDefinition","definition"],["targetUnit","unit"],
    ["targetPopulation","population"],["timeHorizon","time_horizon"],["positiveClass","positive_class"]]) $(id).value = t[key] || "";
  $("targetFamily").value = t.family || "linear";
}
async function persist() {
  await api("bundle", state.bundle);
  state.has_result = false;
  results = null;
  clearResultDisplay();
}
function clearResultDisplay() {
  $("resultEmpty").hidden=false; $("resultContent").hidden=true; $("downloadRun").hidden=true;
}
async function refresh() {
  state = await api("state", undefined, "GET"); csrf = state.csrf;
  if (!state.has_result) clearResultDisplay();
  fillTarget(state.bundle.target);
  $("excludedDatasets").value=(state.bundle.research_policy?.excluded_datasets||[]).join("\n");
  $("connection").textContent = state.connected ? "LLM connected" : "LLM disconnected";
  $("connection").className = "pill" + (state.connected ? " good" : "");
  $("showConnect").textContent = state.connected ? "LLM settings" : "Connect LLM";
  const p = state.profile;
  $("contextInfo").textContent = p ? `${p.rows} rows · ${p.columns.length} columns` : "Nothing uploaded";
  $("testInfo").textContent = state.test_profile ? `${state.test_profile.rows} separate prediction rows` : "Using held-out groups of uploaded rows";
  $("workspaceSummary").textContent = p ? `${p.rows} context rows · ${state.bundle.models.length} candidate models` : "Upload a dataset to begin.";
  $("profileCard").hidden = !p;
  if(p) table($("profileTable"),["Column","Type","Missing","Range"],p.columns.map(c=>[c.name,c.type,c.missing,c.numeric && c.min!==undefined ? `${fmt(c.min)} – ${fmt(c.max)}` : "Review category coding"]));
  $("modelMode").value=state.bundle.model_policy?.mode||"delegate";
  $("modelInstructions").value=state.bundle.model_policy?.instructions||"";
  renderSources(); renderModels(); renderGraph(); renderIssues(); updateJob(state.job);
  if (state.job.status === "running") poll();
}
function fmt(x) { return typeof x === "number" ? Number(x.toPrecision(5)).toLocaleString("en-US", {maximumFractionDigits:4}) : x ?? "unknown"; }
function table(parent, headers, rows, onRow) {
  parent.replaceChildren(); const t=node("table"), head=node("thead"), tr=node("tr");
  headers.forEach(h=>tr.append(node("th",h))); head.append(tr); t.append(head);
  const body=node("tbody"); rows.forEach((row,i)=>{const r=node("tr");row.forEach(v=>r.append(node("td",fmt(v))));if(onRow){r.dataset.row=i;r.onclick=()=>onRow(i);}body.append(r);});
  t.append(body);parent.append(t);
}
function renderSources() {
  const root=$("sources"), sources=state.bundle.sources; root.replaceChildren();root.className=sources.length?"":"empty";
  $("sourceCount").textContent=`${sources.length} sources`;
  if(!sources.length) root.textContent="Find papers automatically or upload your own. Approve the ones you want to use.";
  sources.forEach(s=>{
    const row=node("div",undefined,"source-row"), desc=node("div");
    const decision=s.dataset_review?.decision;
    desc.append(node("h4",s.title),node("p",decision==="exclude"?"Rejected":decision?(s.dataset_review?.actor==="delegated_llm"?"Selected automatically":"Approved"):"Available for automatic selection","small muted"));
    if(s.screening_report?.reason)desc.append(node("p",s.screening_report.reason,"small muted"));
    const details=node("details");details.append(node("summary","Source details"));
    for(const flag of s.screening_flags||[])details.append(node("p",flag,"small muted"));
    if(s.screening_report?.quote)details.append(node("blockquote",s.screening_report.quote));
    const view=node("button","Read source","quiet");view.onclick=()=>viewSource(s);details.append(view);desc.append(details);
    const buttons=node("div"), approve=node("button","Approve","secondary"), reject=node("button","Reject","quiet"), remove=node("button","Remove","quiet danger");
    approve.onclick=action(async()=>{await api("source-review",{source_id:s.id,decision:"approve"});await refresh();});
    reject.onclick=action(async()=>{await api("source-review",{source_id:s.id,decision:"exclude"});await refresh();});
    remove.onclick=action(async()=>{await api("source-remove",{source_id:s.id});await refresh();});
    buttons.append(approve,reject,remove);row.append(desc,buttons);root.append(row);
  });
}
function viewSource(s) {
  $("viewSourceTitle").textContent=s.title;$("viewSourceText").textContent=s.text;
  const a=$("viewSourceLink");a.hidden=!/^https:\/\//i.test(s.url);a.href=a.hidden?"#":s.url;
  $("viewSourceDialog").showModal();
}
function renderIssues() {
  const root=$("researchIssues");root.replaceChildren();const messages=[];
  for(const log of state.research_log||[]){messages.push(...(log.errors||[]),...(log.extraction_issues||[]));}
  for(const item of state.bundle.agent_log||[]){
    if(item.role==="proposer_reviewer"){
      messages.push(`Model proposal/review round ${item.round}`);
      for(const r of item.reviews||[])messages.push(`${r.id}: ${r.accept?"accepted":"needs revision"} — ${r.reason}`);
    }
  }
  const report=state.bundle.preparation;
  if(report){
    messages.push(`${report.accepted} usable equations prepared automatically.`);
    for(const item of report.skipped||[])messages.push(`${item.model_id}: skipped — ${item.reason}`);
    for(const item of report.source_issues||[])messages.push(`${item.source_id}: ${typeof item.reason==="string"?item.reason:JSON.stringify(item.reason)}`);
    for(const [kind,value]of Object.entries(report.coverage||{}))messages.push(`${kind}: ${value.supported}/${value.total} rows supported.`);
  }
  root.hidden=!messages.length;
  if(messages.length){root.append(node("h3","Retrieval & extraction notes"));for(const m of messages)root.append(node("p",typeof m==="string"?m:JSON.stringify(m),"small muted"));}
}
function renderModels() {
  const root=$("models");root.replaceChildren();root.className=state.bundle.models.length?"":"empty";
  if(!state.bundle.models.length)root.textContent="No extracted models yet.";
  state.bundle.models.forEach((m,i)=>{
    const card=node("article",undefined,"model-card"+(m.reviewed?" approved":""));
    const title=node("div",undefined,"card-title");title.append(node("h3",m.title||m.id||"Incomplete draft"),node("span",m.synthetic?"SYNTHETIC DEMO":m.reviewed?(state.bundle.workflow==="automatic"?"Prepared automatically":"Reviewed"):"Not prepared","pill "+(m.reviewed?"good":"warn")));card.append(title);
    if(m.model_origin && m.model_origin!=="published"){
      card.append(node("p",m.model_origin==="user_specified"?"USER-SPECIFIED MODEL — numerical assumptions":"LLM-PROPOSED MODEL — numerical assumptions","pill warn"));
      card.append(node("p",m.proposal_basis||"Assumption-based scenario; not a published fitted equation.","small muted"));
    }
    card.append(node("p",`${m.family||"unknown model"} · ${m.target||"unknown target"} · weight ${fmt(m.weight)} · ${m.source_population||"unknown source population"}`,"muted"));
    const tbody=node("div",undefined,"table-wrap");
    const labels=["Intercept",...(m.predictors||[]).map(p=>p.column||"Unmapped predictor")];
    table(tbody,["Term","Coefficient","Standard error"],labels.map((n,j)=>[n,m.coefficients?.[j],m.standard_errors?.[j]??(m.covariance?.[j]?.[j]>=0?Math.sqrt(m.covariance[j][j]):null)]));card.append(tbody);
    const details=node("details");details.append(node("summary","Mappings, uncertainty & source passages"));
    for(const p of m.predictors||[])details.append(node("p",`${p.column}: ${p.meaning||"meaning unknown"}. ${p.input_unit||"?"} → ${p.study_unit||"?"}; multiplier ${fmt(p.multiplier)}, offset ${fmt(p.offset)}, ${p.transform||"?"}; source center ${fmt(p.source_center)}, scale ${fmt(p.source_scale)}.`,"muted"));
    details.append(node("p",`Residual SD: ${fmt(m.residual_sd)}; log-SD uncertainty: ${fmt(m.residual_log_sd)}; extra transfer-intercept SD: ${fmt(m.transfer_intercept_sd)}.`));
    details.append(node("p",m.uncertainty_basis||"Uncertainty basis missing."),node("p",`Weight justification: ${m.weight_reason||"missing"}`));
    const list=node("ul");(m.assumptions||[]).forEach(x=>list.append(node("li",x)));details.append(list);
    for(const a of m.anchors||[]){details.append(node("blockquote",a.quote||"Missing quote"),node("p",`Source ${a.source_id}: ${a.supports||"support unspecified"}`,"small muted"));}
    card.append(details);
    const actions=node("div",undefined,"model-actions"), approval=node("label",undefined,"check"), check=node("input");check.type="checkbox";check.checked=!!m.reviewed;
    check.onchange=action(async()=>{m.reviewed=check.checked;try{await persist();renderModels();renderGraph();}catch(e){m.reviewed=!check.checked;check.checked=m.reviewed;throw e;}});
    approval.append(check,document.createTextNode("I checked source support, mappings and all stated assumptions."));
    const buttons=node("div"), edit=node("button","Edit JSON","quiet"), remove=node("button","Remove","quiet danger");edit.onclick=()=>editModel(i);
    remove.onclick=action(async()=>{state.bundle.models.splice(i,1);await persist();renderModels();renderGraph();});buttons.append(edit,remove);if(state.bundle.workflow!=="automatic")actions.append(approval);actions.append(buttons);card.append(actions);root.append(card);
  });
}
function svgEl(tag, attrs={}, text) {
  const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;
}
function renderGraph() {
  const root=$("graph");root.replaceChildren();const models=state.bundle.models;
  if(!models.length){root.textContent="Candidate models will connect your predictors to the target here.";return;}
  const columns=[...new Set(models.flatMap(m=>(m.predictors||[]).map(p=>p.column)))].slice(0,8), shown=models.slice(0,6);
  const h=Math.max(columns.length,shown.length,2)*42+30, svg=svgEl("svg",{viewBox:`0 0 680 ${h}`,role:"img","aria-label":"Predictors connected through literature models to target"});
  const positions=arr=>arr.map((_,i)=>25+i*(h-50)/Math.max(arr.length-1,1));const cy=positions(columns),my=positions(shown);
  shown.forEach((m,j)=>{(m.predictors||[]).forEach(p=>{const i=columns.indexOf(p.column);if(i>=0)svg.append(svgEl("path",{d:`M 165 ${cy[i]+12} C 215 ${cy[i]+12},215 ${my[j]+12},265 ${my[j]+12}`}));});svg.append(svgEl("path",{d:`M 435 ${my[j]+12} C 475 ${my[j]+12},475 ${h/2},520 ${h/2}`}));});
  function box(x,y,w,label){svg.append(svgEl("rect",{x,y,width:w,height:28,rx:6}));svg.append(svgEl("text",{x:x+10,y:y+18},String(label).slice(0,23)));}
  columns.forEach((c,i)=>box(5,cy[i]-2,160,c));shown.forEach((m,i)=>box(265,my[i]-2,170,(m.reviewed?"✓ ":"○ ")+(m.id||"Draft")));box(520,h/2-14,155,state.bundle.target.name||"Target");root.append(svg);
}
function editModel(i) {
  editIndex=i;
  const template={id:"new-model",title:"",family:state.bundle.target.family||"linear",target:state.bundle.target.name||"",target_definition:state.bundle.target.definition||"",target_unit:state.bundle.target.unit||"",source_population:"",positive_class:state.bundle.target.positive_class||null,
    predictors:[{column:"",meaning:"",input_unit:"",study_unit:"",multiplier:null,offset:null,transform:"identity",source_center:null,source_scale:null,category_map:null}],
    coefficients:[null,null],covariance:null,standard_errors:[null,null],independent_coefficients:false,parameter_distribution:"multivariate_normal",residual_sd:null,residual_log_sd:null,target_center:null,target_scale:null,target_transform:"identity",transfer_intercept_sd:null,weight:null,weight_reason:"",uncertainty_basis:"",assumptions:["Replace with explicitly justified assumptions"],anchors:[{source_id:state.bundle.sources[0]?.id||"",quote:"",supports:""}],reviewed:false,synthetic:false};
  $("modelJson").value=JSON.stringify(i<0?template:state.bundle.models[i],null,2);$("modelDialog").showModal();
}
function config() {
  return {method:$("method").value,worlds:+$("worlds").value,samples_per_world:+$("samples").value,interval:+$("interval").value,seed:+$("seed").value,
    context_limit:+$("contextLimit").value,folds:+$("folds").value,device:$("device").value,estimators:+$("estimators").value,model_path:$("modelPath").value.trim()||null,
    regression_context_labels:$("contextLabels").value,use_all_features:$("allFeatures").checked,allow_synthetic:$("allowSynthetic").checked,assumptions_approved:true};
}
function updateJob(job) {
  $("status").textContent=job.message;$("progress").value=job.progress||0;$("cancel").hidden=job.status!=="running";
}
async function poll() {
  if(polling)return;polling=true;
  try {
    let data;
    do {
      await new Promise(resolve=>setTimeout(resolve,1000));
      data=await api("job",undefined,"GET");updateJob(data.job);
    } while(data.busy);
    await refresh();
    if(data.job.status==="error"||data.job.status==="cancelled")notify(data.job.message,true);
    else {notify("Operation complete. Review the new sources, models or results.");if(state.has_result)tab("results");else if(state.bundle.preparation)tab("evidence");}
  } catch(e){notify(e.message,true);} finally{polling=false;}
}
async function start(path,body={}) { await api(path,body);$("status").textContent="Working…";poll(); }

$("showConnect").onclick=()=>$("connectDialog").showModal();
$("provider").onchange=()=>$("endpointLabel").hidden=$("provider").value!=="compatible";
$("connectForm").onsubmit=action(async()=>{
  const button=$("connectForm").querySelector('button[type="submit"]');button.disabled=true;
  try{await api("connect",{provider:$("provider").value,endpoint:$("endpoint").value.trim(),model:$("llmModel").value.trim(),token:$("apiToken").value,send_max_tokens:$("maxTokens").checked,share_summaries:$("shareSummaries").checked});$("apiToken").value="";$("connectDialog").close();await refresh();notify("LLM connected. Credentials are held only in this local server session.");}finally{button.disabled=false;}
});
$("disconnect").onclick=action(async()=>{await api("disconnect",{});$("apiToken").value="";$("connectDialog").close();await refresh();});
for(const [id,role]of[["contextFile","context"],["testFile","test"]])$(id).onchange=action(async()=>{const form=new FormData();form.append("file",$(id).files[0]);form.append("role",role);await api("upload",form);await refresh();notify("Dataset loaded. Review predictor meanings and units before approving models.");});
$("clearTest").onclick=action(async()=>{await api("clear-test",{});await refresh();});
$("demo").onclick=action(async()=>{await api("demo",{});await refresh();$("method").value="direct";$("allowSynthetic").checked=true;$("approveRun").checked=false;tab("evidence");notify("Synthetic demo loaded. Review its fictional model, approve it, then choose Prediction settings. No research or LLM request was made.");});
$("saveTarget").onclick=action(async()=>{const target=readTarget();if(JSON.stringify(target)!==JSON.stringify(state.bundle.target))state.bundle.models.forEach(m=>m.reviewed=false);state.bundle.target=target;await persist();renderModels();renderGraph();tab("evidence");notify("Target saved. Find papers, approve sources, then build predictions.");});
$("addSource").onclick=()=>$("sourceDialog").showModal();
$("fetchSource").onclick=action(async()=>{await api("source",{url:$("sourceUrl").value.trim()});$("sourceDialog").close();await refresh();});
$("saveSource").onclick=action(async()=>{await api("source",{title:$("sourceTitle").value.trim(),text:$("sourceText").value});$("sourceDialog").close();await refresh();});
$("paperFile").onchange=action(async()=>{const form=new FormData();form.append("file",$("paperFile").files[0]);await api("source",form);$("sourceDialog").close();await refresh();});
async function savePolicy() {
  await api("research-policy",{excluded_datasets:$("excludedDatasets").value.split("\n").map(x=>x.trim()).filter(Boolean)});
}
function selectedSources(){return state.bundle.sources.filter(s=>s.dataset_review?.decision!=="exclude").slice(0,8).map(s=>s.id);}
$("savePolicy").onclick=action(async()=>{await savePolicy();await refresh();notify("Exclusions saved. Changed exclusions reset source and model approvals.");});
$("taiwanPolicy").onclick=()=>{$("excludedDatasets").value="UCI Default of Credit Card Clients\nDefault of Credit Card Clients\nTaiwan credit card\nYeh and Lien\nYeh & Lien";notify("Taiwan aliases filled in. Click Save exclusions to apply them.");};
$("screenSources").onclick=action(async()=>{const ids=selectedSources();await savePolicy();await start("screen",{source_ids:ids});});

$("research").onclick=action(async()=>{await savePolicy();await start("research");});
$("extract").onclick=action(()=>start("extract",{}));
async function saveModelPolicy(){
  await api("model-policy",{mode:$("modelMode").value,instructions:$("modelInstructions").value});
}
$("saveModelPolicy").onclick=action(async()=>{await saveModelPolicy();await refresh();notify("Model instructions saved.");});
$("suggestModels").onclick=action(async()=>{await saveModelPolicy();await start("suggest-models");});
$("pipeline").onclick=action(async()=>{await saveModelPolicy();await start("pipeline",config());});
$("addModel").onclick=()=>editModel(-1);
$("saveModel").onclick=action(async()=>{const m=JSON.parse($("modelJson").value);m.reviewed=false;
  if(!m.synthetic){m.model_origin="user_specified";m.proposal_basis=m.proposal_basis||"User-specified model: numerical values and uncertainty are user assumptions unless documented otherwise.";state.bundle.model_policy={mode:"delegate",instructions:$("modelInstructions").value};state.bundle.workflow="automatic";}
  if(editIndex<0)state.bundle.models.push(m);else state.bundle.models[editIndex]=m;await persist();$("modelDialog").close();renderModels();renderGraph();notify("Draft saved. Review and approve before validation.");});
$("importEvidence").onchange=action(async()=>{const saved=JSON.parse(await $("importEvidence").files[0].text());const b=saved.bundle?{...saved.bundle,research_log:saved.research_log||[]}:saved;(b.models||[]).forEach(m=>m.reviewed=false);await api("bundle",b);await refresh();$("method").value="tabpfn";tab("evidence");notify("Research imported. Model approvals reset for review against the current dataset.");});
$("validate").onclick=action(async()=>{const r=await api("validate",config());notify(`${r.models} evidence models passed structural, source-quote and data-mapping checks. Scientific correctness still depends on your review.`);});
$("run").onclick=action(()=>start("run",config()));
$("cancel").onclick=action(async()=>notify((await api("cancel",{})).message));
$("chatForm").onsubmit=action(async()=>{const text=$("chatInput").value.trim();if(!text)return;$("chatLog").append(node("div",text,"bubble user"));$("chatInput").value="";$("sendChat").disabled=true;try{const r=await api("chat",{message:text});$("chatLog").append(node("div",r.reply,"bubble assistant"));$("chatLog").scrollTop=$("chatLog").scrollHeight;}finally{$("sendChat").disabled=false;}});

async function loadResults(){
  results=await api("results",undefined,"GET");$("resultEmpty").hidden=true;$("resultContent").hidden=false;$("downloadRun").hidden=false;
  const metrics=$("metrics");metrics.replaceChildren();
  for(const[value,label]of[[results.total_rows,"prediction rows"],[results.audit.config.worlds,"plausible datasets"],[`${Math.round(results.audit.config.interval*100)}%`,"conditional prediction interval"]]){const c=node("div",undefined,"metric");c.append(node("b",value),node("span",label));metrics.append(c);}
  const series=["tabpfn","direct"].filter(s=>results.rows[0]?.[s+"_mean"]!==undefined);
  const headers=["Row",...series.flatMap(s=>[s==="tabpfn"?"Zero-shot mean (TabPFN)":"Diagnostic mean", "Prediction interval"])];
  table($("resultTable"),headers,results.rows.map(r=>[r.row_position,...series.flatMap(s=>[fmt(r[s+"_mean"]),`${fmt(r[s+"_lower"])} – ${fmt(r[s+"_upper"])}`])]),i=>loadDistribution(results.rows[i].row_position).catch(e=>notify(e.message,true)));
  const warnings=$("resultWarnings");warnings.replaceChildren();for(const w of results.audit.warnings)warnings.append(node("p",w));warnings.append(node("p",`Prediction mode: ${results.audit.prediction_mode}. Preview shows at most 200 rows; the download contains every row, components, samples and audit settings.`));
  intervalChart(results.rows.slice(0,30),series);await loadDistribution(results.rows[0].row_position);
}
function intervalChart(rows,series){
  const root=$("intervalChart");root.replaceChildren();const has=series.length>1;
  const values=rows.flatMap(r=>series.flatMap(s=>[r[s+"_lower"],r[s+"_upper"]]));let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=1;hi+=1;}
  const w=720,h=260,x=i=>52+(i+.5)*(w-75)/rows.length,y=v=>h-35-(v-lo)/(hi-lo)*(h-65),svg=svgEl("svg",{viewBox:`0 0 ${w} ${h}`,role:"img","aria-label":"Prediction intervals by row"});
  for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,yy=y(v);svg.append(svgEl("line",{x1:52,y1:yy,x2:w-20,y2:yy,stroke:"#ffffff14"}),svgEl("text",{x:2,y:yy+4,fill:"#8b8d9e","font-size":10},fmt(v)));}
  rows.forEach((r,i)=>{series.forEach((s,j)=>{const xx=x(i)+(has?(j?3:-3):0),color=s==="direct"?"#8078c8":"#6db5c4";svg.append(svgEl("line",{x1:xx,x2:xx,y1:y(r[s+"_lower"]),y2:y(r[s+"_upper"]),stroke:color,"stroke-width":2}));svg.append(svgEl("circle",{cx:xx,cy:y(r[s+"_mean"]),r:3,fill:color}));});if(i%Math.max(1,Math.ceil(rows.length/10))===0)svg.append(svgEl("text",{x:x(i),y:h-10,fill:"#8b8d9e","font-size":10},String(r.row_position)));});root.append(svg);
}
async function loadDistribution(row){
  const data=await api("distribution/"+row,undefined,"GET"),root=$("distributionChart");root.replaceChildren();$("distTitle").textContent=`Predictive distribution · row ${row}`;
  const w=720,h=240,svg=svgEl("svg",{viewBox:`0 0 ${w} ${h}`,role:"img","aria-label":"Predictive distribution of the selected output"}),names=Object.keys(data.series),max=Math.max(1,...Object.values(data.series).flat());
  names.forEach(name=>data.series[name].forEach((n,i)=>{const bw=(w-70)/40;svg.append(svgEl("rect",{x:50+i*bw,y:h-30-(n/max)*(h-55),width:bw-.8,height:(n/max)*(h-55),fill:name.startsWith("direct")?"#8078c8":"#6db5c4",opacity:.5}));}));
  for(let k=0;k<=4;k++){const i=k*10;svg.append(svgEl("text",{x:50+i*(w-70)/40,y:h-8,fill:"#8b8d9e","font-size":10,"text-anchor":"middle"},fmt(data.edges[i])));}root.append(svg);
}
refresh().catch(e=>notify(e.message,true));
