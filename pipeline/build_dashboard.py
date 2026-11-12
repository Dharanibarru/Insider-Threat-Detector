"""
build_dashboard.py
--------------------
Builds the self-contained, offline SOC-style dashboard (results/dashboard.html).

Reads directly from the model output (../data/predictions.csv) — no
intermediate files needed. Run this AFTER anomaly_detection.py.
"""

import json
import pandas as pd

# ---- Load model predictions and build the two compact JSON payloads ----
df = pd.read_csv("../data/predictions.csv", parse_dates=["date"])

df["t"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
df["hr"] = df["login_hour"].round(1)
df["dur"] = df["session_duration_hrs"].round(2)
df["risk"] = df["risk_score"].round(1)

sessions = df.rename(columns={
    "is_new_device": "newdev", "is_off_hours": "offhr",
    "sessions_today": "freq", "iso_pred": "flag", "is_malicious_gt": "gt"
})[["user", "t", "pc", "hr", "dur", "newdev", "offhr", "freq", "risk", "flag", "gt"]]

users = df.groupby("user").agg(
    total_sessions=("risk_score", "count"),
    avg_risk=("risk_score", "mean"),
    max_risk=("risk_score", "max"),
    flagged_sessions=("iso_pred", "sum"),
    is_malicious_gt=("is_malicious_gt", "max"),
).reset_index().round(1)
users.to_csv("../data/user_risk_summary.csv", index=False)

sessions_json = json.dumps(sessions.to_dict(orient="records"), separators=(",", ":"))
users_json = json.dumps(users.to_dict(orient="records"), separators=(",", ":"))

html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SENTINEL — Insider Threat Console</title>
<style>
  :root{
    --bg: #0a0d12;
    --panel: #10141c;
    --panel-2: #141924;
    --border: #232937;
    --text: #d7dce5;
    --text-dim: #7b8496;
    --text-faint: #4a5164;
    --cyan: #4fd8e0;
    --amber: #e0a94f;
    --red: #e0554f;
    --green: #4fe08f;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .mono{font-family:'SFMono-Regular','Consolas','Menlo',monospace;}

  .topbar{
    display:flex; align-items:center; justify-content:space-between;
    padding:16px 24px; border-bottom:1px solid var(--border);
    background:linear-gradient(180deg, #0d1119, #0a0d12);
  }
  .brand{display:flex; align-items:center; gap:10px;}
  .brand .dot{width:9px; height:9px; border-radius:50%; background:var(--cyan); box-shadow:0 0 10px var(--cyan); animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}
  .brand h1{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:15px; letter-spacing:.14em; margin:0; font-weight:700; color:#eef1f6;}
  .brand span{color:var(--text-dim); font-size:11px; font-family:'SFMono-Regular','Consolas','Menlo',monospace; letter-spacing:.08em;}

  .stats{display:flex; gap:28px;}
  .stat{text-align:right;}
  .stat .val{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:20px; font-weight:600; color:#eef1f6;}
  .stat .lbl{font-size:10px; text-transform:uppercase; letter-spacing:.1em; color:var(--text-dim); margin-top:2px;}
  .stat.alert .val{color:var(--red);}

  .layout{display:grid; grid-template-columns: 380px 1fr; gap:1px; background:var(--border); min-height:calc(100vh - 62px);}
  .panel{background:var(--panel);}

  .panel-head{
    padding:14px 18px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between;
  }
  .panel-head h2{font-size:11px; text-transform:uppercase; letter-spacing:.12em; color:var(--text-dim); margin:0; font-weight:600;}
  .panel-head .count{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:11px; color:var(--text-faint);}

  .userlist{max-height:calc(100vh - 110px); overflow-y:auto;}
  .urow{
    display:flex; align-items:center; gap:12px; padding:10px 18px;
    border-bottom:1px solid #171c27; cursor:pointer; transition:background .12s;
  }
  .urow:hover{background:var(--panel-2);}
  .urow.active{background:#182130; border-left:3px solid var(--cyan); padding-left:15px;}
  .urow .uid{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:12.5px; color:#eef1f6; flex:1;}
  .urow .uid .tag{font-size:9px; color:var(--red); border:1px solid var(--red); border-radius:3px; padding:1px 4px; margin-left:6px; letter-spacing:.05em;}
  .riskbar-track{width:64px; height:5px; background:#1c2230; border-radius:3px; overflow:hidden;}
  .riskbar-fill{height:100%; border-radius:3px;}
  .riskval{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:11px; width:32px; text-align:right; color:var(--text-dim);}

  .main{padding:22px 26px; overflow-y:auto; max-height:calc(100vh - 62px);}
  .empty-state{color:var(--text-faint); font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:13px; padding:60px 0; text-align:center;}

  .detail-head{display:flex; align-items:baseline; justify-content:space-between; margin-bottom:18px; flex-wrap:wrap; gap:10px;}
  .detail-head h2{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:22px; margin:0; color:#f4f6fa; font-weight:600;}
  .badge{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:10.5px; letter-spacing:.08em; padding:4px 9px; border-radius:4px; text-transform:uppercase; font-weight:600;}
  .badge.high{background:rgba(224,85,79,.15); color:var(--red); border:1px solid rgba(224,85,79,.4);}
  .badge.med{background:rgba(224,169,79,.15); color:var(--amber); border:1px solid rgba(224,169,79,.4);}
  .badge.low{background:rgba(79,224,143,.13); color:var(--green); border:1px solid rgba(79,224,143,.35);}

  .metrics-row{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:22px;}
  .mcard{background:var(--panel-2); border:1px solid var(--border); border-radius:6px; padding:12px 14px;}
  .mcard .mlbl{font-size:10px; text-transform:uppercase; letter-spacing:.09em; color:var(--text-dim); margin-bottom:6px;}
  .mcard .mval{font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:19px; color:#eef1f6; font-weight:600;}

  .chart-wrap{background:var(--panel-2); border:1px solid var(--border); border-radius:6px; padding:16px 18px 8px; margin-bottom:22px;}
  .chart-title{font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--text-dim); margin-bottom:10px; font-weight:600;}
  svg{display:block;}

  table{width:100%; border-collapse:collapse; font-family:'SFMono-Regular','Consolas','Menlo',monospace; font-size:12px;}
  thead th{
    text-align:left; padding:8px 10px; color:var(--text-dim); font-weight:600;
    text-transform:uppercase; font-size:10px; letter-spacing:.07em;
    border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--panel);
  }
  tbody td{padding:7px 10px; border-bottom:1px solid #161b26; color:var(--text);}
  tbody tr:hover{background:var(--panel-2);}
  .flagrow{background:rgba(224,85,79,.06);}
  .pill{display:inline-block; padding:1px 7px; border-radius:3px; font-size:10px; font-weight:600;}
  .pill.yes{background:rgba(224,85,79,.18); color:var(--red);}
  .pill.no{color:var(--text-faint);}

  .legend{display:flex; gap:16px; margin-top:8px; font-size:10.5px; color:var(--text-dim);}
  .legend span{display:inline-flex; align-items:center; gap:5px;}
  .legend i{width:9px; height:9px; border-radius:2px; display:inline-block;}

  ::-webkit-scrollbar{width:8px; height:8px;}
  ::-webkit-scrollbar-track{background:var(--bg);}
  ::-webkit-scrollbar-thumb{background:#232937; border-radius:4px;}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="dot"></div>
    <div>
      <h1>SENTINEL <span style="color:var(--text-faint); font-weight:400;">/</span> <span style="font-weight:400; color:var(--text-dim);">insider threat console</span></h1>
      <span>login &amp; session behavioral anomaly detection · isolation forest</span>
    </div>
  </div>
  <div class="stats" id="topstats"></div>
</div>

<div class="layout">
  <div class="panel">
    <div class="panel-head">
      <h2>Monitored Users</h2>
      <span class="count" id="ucount"></span>
    </div>
    <div class="userlist" id="userlist"></div>
  </div>

  <div class="panel main" id="main">
    <div class="empty-state">Select a user from the list to inspect session-level risk detail →</div>
  </div>
</div>

<script id="sessions-data" type="application/json">__SESSIONS_JSON__</script>
<script id="users-data" type="application/json">__USERS_JSON__</script>

<script>
const SESSIONS = JSON.parse(document.getElementById('sessions-data').textContent);
const USERS = JSON.parse(document.getElementById('users-data').textContent);

function riskColor(v){
  if(v >= 70) return getComputedStyle(document.documentElement).getPropertyValue('--red');
  if(v >= 40) return getComputedStyle(document.documentElement).getPropertyValue('--amber');
  return getComputedStyle(document.documentElement).getPropertyValue('--green');
}
function riskBadgeClass(v){ return v>=70?'high':(v>=40?'med':'low'); }
function riskLabel(v){ return v>=70?'High Risk':(v>=40?'Elevated':'Normal'); }

// ---- Top stats ----
const totalUsers = USERS.length;
const flaggedUsers = USERS.filter(u => u.max_risk >= 70).length;
const totalSessions = SESSIONS.length;
const totalAlerts = SESSIONS.filter(s => s.flag === 1).length;

document.getElementById('topstats').innerHTML = `
  <div class="stat"><div class="val">${totalUsers}</div><div class="lbl">Monitored Users</div></div>
  <div class="stat"><div class="val">${totalSessions.toLocaleString()}</div><div class="lbl">Sessions Analyzed</div></div>
  <div class="stat"><div class="val">${totalAlerts}</div><div class="lbl">Anomaly Flags</div></div>
  <div class="stat alert"><div class="val">${flaggedUsers}</div><div class="lbl">High-Risk Users</div></div>
`;
document.getElementById('ucount').textContent = totalUsers + ' total';

// ---- User list ----
const sorted = [...USERS].sort((a,b) => b.max_risk - a.max_risk);
const listEl = document.getElementById('userlist');
sorted.forEach(u => {
  const row = document.createElement('div');
  row.className = 'urow';
  row.dataset.user = u.user;
  row.innerHTML = `
    <div class="uid">${u.user}${u.is_malicious_gt ? '<span class="tag">GT</span>' : ''}</div>
    <div class="riskbar-track"><div class="riskbar-fill" style="width:${u.max_risk}%; background:${riskColor(u.max_risk)};"></div></div>
    <div class="riskval">${u.max_risk.toFixed(0)}</div>
  `;
  row.addEventListener('click', () => selectUser(u.user));
  listEl.appendChild(row);
});

function selectUser(userId){
  document.querySelectorAll('.urow').forEach(r => r.classList.toggle('active', r.dataset.user === userId));
  const u = USERS.find(x => x.user === userId);
  const sess = SESSIONS.filter(s => s.user === userId).sort((a,b) => a.t.localeCompare(b.t));

  const main = document.getElementById('main');
  const badgeClass = riskBadgeClass(u.max_risk);

  let html = `
    <div class="detail-head">
      <h2>${u.user}</h2>
      <span class="badge ${badgeClass}">${riskLabel(u.max_risk)} · peak ${u.max_risk.toFixed(0)}/100</span>
    </div>
    <div class="metrics-row">
      <div class="mcard"><div class="mlbl">Total Sessions</div><div class="mval">${u.total_sessions}</div></div>
      <div class="mcard"><div class="mlbl">Avg Risk Score</div><div class="mval">${u.avg_risk.toFixed(1)}</div></div>
      <div class="mcard"><div class="mlbl">Peak Risk Score</div><div class="mval">${u.max_risk.toFixed(1)}</div></div>
      <div class="mcard"><div class="mlbl">Flagged Sessions</div><div class="mval">${u.flagged_sessions}</div></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Risk Score Timeline</div>
      ${buildSparkline(sess)}
      <div class="legend">
        <span><i style="background:var(--green)"></i>Normal (&lt;40)</span>
        <span><i style="background:var(--amber)"></i>Elevated (40-70)</span>
        <span><i style="background:var(--red)"></i>High Risk (&ge;70)</span>
      </div>
    </div>
    <div class="chart-title" style="margin-bottom:8px;">Session Log</div>
    <div style="max-height:360px; overflow-y:auto; border:1px solid var(--border); border-radius:6px;">
    <table>
      <thead><tr>
        <th>Timestamp</th><th>PC</th><th>Login Hr</th><th>Duration (h)</th>
        <th>New Device</th><th>Off-Hours</th><th>Sessions/Day</th><th>Risk</th><th>Flag</th>
      </tr></thead>
      <tbody>
        ${sess.map(s => `
          <tr class="${s.flag ? 'flagrow' : ''}">
            <td>${s.t}</td>
            <td>${s.pc}</td>
            <td>${s.hr.toFixed(1)}</td>
            <td>${s.dur.toFixed(2)}</td>
            <td>${s.newdev ? '<span class="pill yes">NEW</span>' : '<span class="pill no">—</span>'}</td>
            <td>${s.offhr ? '<span class="pill yes">YES</span>' : '<span class="pill no">—</span>'}</td>
            <td>${s.freq}</td>
            <td style="color:${riskColor(s.risk)}; font-weight:600;">${s.risk.toFixed(1)}</td>
            <td>${s.flag ? '<span class="pill yes">ALERT</span>' : '<span class="pill no">—</span>'}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
    </div>
  `;
  main.innerHTML = html;
}

function buildSparkline(sess){
  const w = 900, h = 130, pad = 6;
  if(sess.length === 0) return '<div style="color:var(--text-faint); font-size:12px;">No session data.</div>';
  const maxR = 100;
  const stepX = (w - pad*2) / Math.max(sess.length - 1, 1);
  let path = '';
  let dots = '';
  sess.forEach((s, i) => {
    const x = pad + i * stepX;
    const y = h - pad - (s.risk / maxR) * (h - pad*2);
    path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
    dots += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${s.flag ? 3.2 : 1.6}" fill="${riskColor(s.risk)}" opacity="${s.flag ? 1 : 0.55}"/>`;
  });
  // threshold lines
  const yAt = v => h - pad - (v/maxR)*(h-pad*2);
  return `
    <svg viewBox="0 0 ${w} ${h}" style="width:100%; height:${h}px;">
      <line x1="0" y1="${yAt(40)}" x2="${w}" y2="${yAt(40)}" stroke="#2a3140" stroke-dasharray="3,3"/>
      <line x1="0" y1="${yAt(70)}" x2="${w}" y2="${yAt(70)}" stroke="#2a3140" stroke-dasharray="3,3"/>
      <path d="${path}" fill="none" stroke="#3a4256" stroke-width="1.3"/>
      ${dots}
    </svg>
  `;
}

// auto-select first (highest-risk) user
if(sorted.length) selectUser(sorted[0].user);
</script>

</body>
</html>
"""

html = html_template.replace("__SESSIONS_JSON__", sessions_json).replace("__USERS_JSON__", users_json)

with open('../results/dashboard.html', 'w') as f:
    f.write(html)

print("Dashboard written:", len(html), "bytes")
