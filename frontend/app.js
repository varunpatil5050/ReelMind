// ReelMind — Feed Client
// Fetches ML recommendations, renders cards, sends interaction events back.

const API = "http://localhost:8001";
let USER_ID = "u_100";

const VIDEOS = [
    "https://www.w3schools.com/html/mov_bbb.mp4",
    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm"
];

const POSTERS = [
    "/static/assets/video_mockup_car_1781987700395.png",
    "/static/assets/video_mockup_city_1781987710435.png",
    "/static/assets/video_mockup_dance_1781987722955.png",
    "/static/assets/video_mockup_portal_1781987732502.png"
];

const CAPTIONS = [
    "This is what happens when the algorithm actually works 🔥",
    "POV: your recommendation engine just hit 50ms inference",
    "When the Two-Tower model finds your vibe ✨",
    "FAISS retrieval hitting different today 💯",
    "The DeepFM ranker said this is for you 🎯",
    "Thompson Sampling chose this. Trust the process.",
    "Your embeddings are showing 📊",
    "Cold start? Never heard of her.",
    "This video's score was 0.94. You're welcome.",
    "The RL optimizer diversified your feed 🌈",
];

const feedContainer = document.getElementById("feed-container");
const template = document.getElementById("video-card-template");
const toast = document.getElementById("personalization-toast");

let feedData = [];
let likeState = {}; // track which videos are liked

// ─── Fetch Feed ─────────────────────────────────────

async function loadFeed() {
    // Show loading
    feedContainer.innerHTML = `
        <div class="loading-card">
            <div class="spinner"></div>
            <span>Loading recommendations...</span>
        </div>
    `;

    try {
        const res = await fetch(`${API}/v1/feed`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: USER_ID, num_results: 10 })
        });
        const data = await res.json();
        feedData = data.videos || [];
        renderFeed(feedData);
    } catch (err) {
        feedContainer.innerHTML = `
            <div class="loading-card">
                <span>⚠️ Could not reach API. Are the servers running?</span>
                <span style="font-size:12px; color: rgba(255,255,255,0.4);">${err.message}</span>
            </div>
        `;
    }
}

// ─── Render Feed ────────────────────────────────────

function renderFeed(videos) {
    feedContainer.innerHTML = "";

    videos.forEach((video, i) => {
        const clone = template.content.cloneNode(true);
        const card = clone.querySelector(".video-card");

        // Video
        const videoEl = card.querySelector(".video-player");
        videoEl.src = VIDEOS[i % VIDEOS.length];
        videoEl.poster = POSTERS[i % POSTERS.length];

        // Creator
        const creatorNum = video.video_id.split("_")[1] || i;
        card.querySelector(".creator-name").textContent = `@creator_${creatorNum}`;

        // Category tag
        const catTag = card.querySelector(".video-category-tag");
        catTag.textContent = video.category || "general";

        // Caption
        card.querySelector(".video-caption").textContent = CAPTIONS[i % CAPTIONS.length];

        // Like count (random for realism)
        const likeCount = Math.floor(Math.random() * 50000) + 500;
        card.querySelector(".like-count").textContent = formatCount(likeCount);

        // ML debug info
        card.querySelector(".score-tag").textContent = `Score: ${(video.score * 100).toFixed(1)}%`;
        card.querySelector(".rank-tag").textContent = `Rank: #${video.rank}`;
        card.querySelector(".source-tag").textContent = video.retrieval_source;

        // Store metadata on card
        card.dataset.videoId = video.video_id;
        card.dataset.category = video.category || "general";
        card.dataset.index = i;

        // Wire interaction buttons
        wireActions(card, video);

        feedContainer.appendChild(card);
    });

    // Start autoplay observer
    setupAutoplay();
}

// ─── Wire Interaction Buttons ───────────────────────

function wireActions(card, video) {
    const likeBtn = card.querySelector(".like-btn");
    const shareBtn = card.querySelector(".share-btn");
    const commentBtn = card.querySelector(".comment-btn");

    likeBtn.addEventListener("click", () => {
        const isLiked = likeBtn.classList.toggle("liked");
        sendEvent(video.video_id, card.dataset.category, isLiked ? "like" : "watch");
        if (isLiked) showToast("Liked! Affinities updating...");
    });

    shareBtn.addEventListener("click", () => {
        sendEvent(video.video_id, card.dataset.category, "share");
        showToast("Shared! Boosting similar content...");
    });

    commentBtn.addEventListener("click", () => {
        sendEvent(video.video_id, card.dataset.category, "comment");
    });
}

// ─── Send Event to Backend ──────────────────────────

async function sendEvent(videoId, category, eventType, watchDuration = 0, watchPct = 0) {
    try {
        await fetch(`${API}/v1/events`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: USER_ID,
                video_id: videoId,
                event_type: eventType,
                category: category,
                watch_duration_ms: watchDuration,
                watch_percentage: watchPct,
            })
        });
    } catch (e) {
        // Silent fail — events are non-critical
    }
}

// ─── Autoplay with IntersectionObserver ─────────────

function setupAutoplay() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target.querySelector(".video-player");
            const card = entry.target;
            if (!video) return;

            if (entry.isIntersecting) {
                video.play().catch(() => {});
                // Track watch event after 2 seconds
                card._watchTimer = setTimeout(() => {
                    const pct = video.currentTime / Math.max(video.duration, 1);
                    sendEvent(
                        card.dataset.videoId,
                        card.dataset.category,
                        "watch",
                        video.currentTime * 1000,
                        Math.min(pct, 1.0)
                    );
                }, 2000);
            } else {
                video.pause();
                if (card._watchTimer) clearTimeout(card._watchTimer);

                // If user scrolled past quickly = skip
                const pct = video.currentTime / Math.max(video.duration, 1);
                if (pct < 0.25 && video.currentTime > 0.3) {
                    sendEvent(card.dataset.videoId, card.dataset.category, "skip", video.currentTime * 1000, pct);
                }
            }
        });
    }, { threshold: 0.7 });

    document.querySelectorAll(".video-card").forEach(card => observer.observe(card));
}

// ─── Helpers ────────────────────────────────────────

function formatCount(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "K";
    return n.toString();
}

function showToast(msg) {
    toast.querySelector(".toast-text").textContent = msg;
    toast.classList.remove("hidden");
    toast.classList.add("visible");
    setTimeout(() => {
        toast.classList.remove("visible");
        toast.classList.add("hidden");
    }, 2000);
}

// ─── User Switcher + Cold Start ─────────────────────

const userSelect = document.getElementById("user-select");
const coldStartBtn = document.getElementById("btn-cold-start");

userSelect.addEventListener("change", () => {
    USER_ID = userSelect.value;
    showToast(`Switched to ${USER_ID}. Loading personalized feed...`);
    loadFeed();
});

coldStartBtn.addEventListener("click", () => {
    const newId = `u_${Date.now() % 100000}`;
    // Add to dropdown
    const option = document.createElement("option");
    option.value = newId;
    option.textContent = `New User #${newId.split("_")[1]} (cold)`;
    option.selected = true;
    userSelect.appendChild(option);
    USER_ID = newId;
    showToast(`Cold start: ${newId}. Zero history — pure exploration!`);
    loadFeed();
});

// ─── Boot ───────────────────────────────────────────

loadFeed();
