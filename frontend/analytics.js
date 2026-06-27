// analytics.js — Fetches user profile from API and renders analytics.

// analytics.js — Fetches user profile from API and renders analytics.

const API = "http://localhost:8001";
let USER_ID = "u_100";

async function fetchProfile() {
    try {
        const res = await fetch(`${API}/v1/user/${USER_ID}/profile`);
        const data = await res.json();
        
        document.getElementById("uid").textContent = USER_ID;
        
        renderSessionStats(data.session_stats);
        renderAffinities(data.category_affinities);
        renderEventLog(data.recent_events);
        renderPipelineTimings(data.last_feed_timings);
        renderFeedExplanation(data.last_feed);
    } catch (err) {
        console.error("Failed to fetch profile:", err);
    }
}

// ─── User Switcher + Cold Start ─────────────────────

const userSelect = document.getElementById("user-select");
const coldStartBtn = document.getElementById("btn-cold-start");

if (userSelect) {
    userSelect.addEventListener("change", () => {
        USER_ID = userSelect.value;
        fetchProfile();
    });
}

if (coldStartBtn) {
    coldStartBtn.addEventListener("click", () => {
        const newId = `u_${Date.now() % 100000}`;
        const option = document.createElement("option");
        option.value = newId;
        option.textContent = `New User #${newId.split("_")[1]} (cold)`;
        option.selected = true;
        userSelect.appendChild(option);
        USER_ID = newId;
        fetchProfile();
    });
}

// ─── Session Stats ──────────────────────────────────

function renderSessionStats(stats) {
    const el = document.getElementById("session-stats");
    if (!stats || !stats.total_watches) {
        el.innerHTML = `<div class="empty-state"><div class="emoji">📊</div>Interact with the feed first</div>`;
        return;
    }

    const rows = [
        ["Videos Watched", stats.total_watches, ""],
        ["Likes", stats.total_likes, "green"],
        ["Skips", stats.total_skips, "red"],
        ["Shares", stats.total_shares, "amber"],
        ["Avg Watch Time", (stats.avg_watch_time_ms / 1000).toFixed(1) + "s", ""],
        ["Avg Watch %", (stats.avg_watch_percentage * 100).toFixed(1) + "%", ""],
        ["Retention Score", stats.retention_score.toFixed(3), stats.retention_score > 0.5 ? "green" : "red"],
        ["Engagement Rate", (stats.engagement_rate * 100).toFixed(2) + "%", stats.engagement_rate > 0.1 ? "green" : "amber"],
        ["Session Duration", stats.session_duration_s.toFixed(0) + "s", ""],
    ];

    el.innerHTML = rows.map(([label, value, color]) =>
        `<div class="stat-row">
            <span class="stat-label">${label}</span>
            <span class="stat-value ${color}">${value}</span>
        </div>`
    ).join("");
}

// ─── Category Affinities ────────────────────────────

function renderAffinities(affinities) {
    const el = document.getElementById("affinities");
    if (!affinities || Object.keys(affinities).length === 0) {
        el.innerHTML = `<div class="empty-state"><div class="emoji">🧠</div>Like/skip videos to build affinities</div>`;
        return;
    }

    el.innerHTML = Object.entries(affinities).map(([cat, score]) =>
        `<div class="affinity-item">
            <div class="affinity-header">
                <span class="affinity-name">${cat}</span>
                <span class="affinity-score">${score.toFixed(3)}</span>
            </div>
            <div class="affinity-bar-bg">
                <div class="affinity-bar-fill" style="width: ${score * 100}%"></div>
            </div>
        </div>`
    ).join("");
}

// ─── Event Log ──────────────────────────────────────

function renderEventLog(events) {
    const el = document.getElementById("event-log");
    if (!events || events.length === 0) {
        el.innerHTML = `<div class="empty-state"><div class="emoji">📝</div>No events yet</div>`;
        return;
    }

    // Show newest first
    const reversed = [...events].reverse();
    el.innerHTML = reversed.map(e => {
        const time = new Date(e.timestamp * 1000).toLocaleTimeString();
        return `<div class="event-entry">
            <span class="event-type-${e.event_type}">${e.event_type.toUpperCase()}</span>
            ${e.video_id} · ${e.category} · ${time}
        </div>`;
    }).join("");
}

// ─── Pipeline Timings ───────────────────────────────

function renderPipelineTimings(timings) {
    const el = document.getElementById("pipeline-timings");
    if (!timings) {
        el.innerHTML = `<div class="empty-state"><div class="emoji">⚡</div>Load the feed to see pipeline timings</div>`;
        return;
    }

    const total = timings.total_ms || 1;
    const stages = [
        ["Feature Fetch", (timings.user_features_ms || 0) + (timings.item_features_ms || 0), "bar-purple"],
        ["FAISS Retrieval", timings.retrieval_ms || 0, "bar-green"],
        ["LightGBM PreRank", timings.prerank_ms || 0, "bar-amber"],
        ["DeepFM HeavyRank", timings.heavy_rank_ms || 0, "bar-red"],
        ["RL Optimizer", timings.rl_ms || 0, "bar-blue"],
    ];

    el.innerHTML = stages.map(([label, ms, barClass]) => {
        const pct = Math.max((ms / total) * 100, 3);
        return `<div class="pipeline-row">
            <span class="pipeline-label">${label}</span>
            <div class="pipeline-bar-bg">
                <div class="pipeline-bar-fill ${barClass}" style="width: ${pct}%">${ms.toFixed(1)}ms</div>
            </div>
        </div>`;
    }).join("") + `
        <div class="stat-row" style="margin-top:12px; border-top: 1px solid #1a1a1a; padding-top:10px;">
            <span class="stat-label">Total E2E Latency</span>
            <span class="stat-value green">${total.toFixed(1)}ms</span>
        </div>
    `;
}

// ─── Feed Explanation ───────────────────────────────

function renderFeedExplanation(feed) {
    const el = document.getElementById("feed-explanation");
    if (!feed || feed.length === 0) {
        el.innerHTML = `<div class="empty-state"><div class="emoji">🎯</div>Load the feed to see recommendation explanations</div>`;
        return;
    }

    el.innerHTML = `
        <table class="feed-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Video ID</th>
                    <th>Category</th>
                    <th>Score</th>
                    <th>Source</th>
                </tr>
            </thead>
            <tbody>
                ${feed.map(v => `
                    <tr>
                        <td>#${v.rank}</td>
                        <td>${v.video_id}</td>
                        <td><span class="tag tag-category">${v.category || '—'}</span></td>
                        <td>${(v.score * 100).toFixed(2)}%</td>
                        <td>${v.retrieval_source}</td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;
}

// ─── Auto-refresh ───────────────────────────────────

fetchProfile();
setInterval(fetchProfile, 3000);
