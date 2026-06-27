// dashboard.js

const METRICS_URL = "http://localhost:8001/metrics";
const FEED_URL = "http://localhost:8001/v1/feed";

// 1. Line Chart (Prometheus Overall Latency)
const ctxLatency = document.getElementById('latencyChart').getContext('2d');
const latencyChart = new Chart(ctxLatency, {
    type: 'line',
    data: {
        labels: [], 
        datasets: [{
            label: 'Gateway Prometheus Cumulative Avg Latency (ms)',
            data: [],
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#c9d1d9' } } },
        scales: {
            x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
            y: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' }, beginAtZero: true }
        }
    }
});

// 2. Stacked Bar Chart (Microservice Pipeline Breakdown)
const ctxPipeline = document.getElementById('pipelineChart').getContext('2d');
const pipelineChart = new Chart(ctxPipeline, {
    type: 'bar',
    data: {
        labels: [], // Request IDs or iterations
        datasets: [
            { label: 'Feature Fetching', data: [], backgroundColor: '#8957e5' }, // purple
            { label: 'FAISS Retrieval', data: [], backgroundColor: '#3fb950' }, // green
            { label: 'LightGBM PreRank', data: [], backgroundColor: '#ffa657' }, // orange
            { label: 'DeepFM HeavyRank', data: [], backgroundColor: '#ff7b72' }, // red
            { label: 'RL Optimizer', data: [], backgroundColor: '#a5d6ff' }, // light blue
            { label: 'Network/Overhead', data: [], backgroundColor: '#30363d' } // dark grey
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: { display: true, text: 'Microservice Latency Breakdown per Request (ms)', color: '#c9d1d9', font: { size: 16 } },
            legend: { labels: { color: '#c9d1d9' } },
            tooltip: { mode: 'index', intersect: false }
        },
        scales: {
            x: { stacked: true, ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
            y: { stacked: true, ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
        }
    }
});

// Fetch overall metrics periodically
async function fetchPrometheusMetrics() {
    try {
        const response = await fetch(METRICS_URL);
        const text = await response.text();

        let feedReqCount = 0;
        let feedReqSum = 0;

        const lines = text.split('\n');
        for (const line of lines) {
            if (line.startsWith('reelmind_gateway_requests_total') && line.includes('endpoint="/v1/feed"')) {
                feedReqCount += parseFloat(line.split(' ')[1]);
            }
            if (line.startsWith('reelmind_gateway_request_latency_seconds_sum') && line.includes('endpoint="/v1/feed"')) {
                feedReqSum += parseFloat(line.split(' ')[1]);
            }
        }

        let avgLatencyMs = 0;
        if (feedReqCount > 0) {
            avgLatencyMs = (feedReqSum / feedReqCount) * 1000;
        }

        const now = new Date();
        const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0') + ':' + String(now.getSeconds()).padStart(2, '0');
        
        latencyChart.data.labels.push(timeStr);
        latencyChart.data.datasets[0].data.push(avgLatencyMs);

        if (latencyChart.data.labels.length > 20) {
            latencyChart.data.labels.shift();
            latencyChart.data.datasets[0].data.shift();
        }

        latencyChart.update();
    } catch (err) {
        console.error("Error fetching prometheus metrics:", err);
    }
}
setInterval(fetchPrometheusMetrics, 2000);
fetchPrometheusMetrics();

// 3. Inference Simulation Logic
const btnSimulate = document.getElementById('btn-simulate');
let simulationCount = 0;

async function runSingleInference() {
    const t0 = performance.now();
    try {
        const response = await fetch(FEED_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: "u_999", num_results: 15 })
        });
        const data = await response.json();
        
        const realTotalMs = performance.now() - t0;
        const timings = data.debug_timings;

        // Parse breakdown
        const featMs = (timings.user_features_ms || 0) + (timings.item_features_ms || 0);
        const retMs = timings.retrieval_ms || 0;
        const prerankMs = timings.prerank_ms || 0;
        const heavyMs = timings.heavy_rank_ms || 0;
        const rlMs = timings.rl_ms || 0;
        
        const sumTracked = featMs + retMs + prerankMs + heavyMs + rlMs;
        const overheadMs = Math.max(0, timings.total_ms - sumTracked);

        simulationCount++;
        const label = `Req #${simulationCount}`;

        pipelineChart.data.labels.push(label);
        pipelineChart.data.datasets[0].data.push(featMs);
        pipelineChart.data.datasets[1].data.push(retMs);
        pipelineChart.data.datasets[2].data.push(prerankMs);
        pipelineChart.data.datasets[3].data.push(heavyMs);
        pipelineChart.data.datasets[4].data.push(rlMs);
        pipelineChart.data.datasets[5].data.push(overheadMs);

        // Keep last 15 simulations
        if (pipelineChart.data.labels.length > 15) {
            pipelineChart.data.labels.shift();
            pipelineChart.data.datasets.forEach(ds => ds.data.shift());
        }

        pipelineChart.update();

    } catch (err) {
        console.error("Simulation failed:", err);
    }
}

btnSimulate.addEventListener('click', async () => {
    btnSimulate.disabled = true;
    btnSimulate.innerText = "Simulating Traffic...";
    
    // Run 10 sequential requests
    for (let i = 0; i < 10; i++) {
        await runSingleInference();
        // tiny sleep between requests to not overwhelm local mac
        await new Promise(r => setTimeout(r, 200)); 
    }
    
    btnSimulate.innerText = "Run Inference Simulation (x10)";
    btnSimulate.disabled = false;
});
