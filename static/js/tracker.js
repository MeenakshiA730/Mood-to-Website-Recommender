(function() {
  // Simple client-side tracker for content page

  // Capture timings
  let sessionStart = Date.now();
  let maxScroll = 0;

  function getScrollDepth() {
    const doc = document.documentElement;
    const body = document.body;
    const scrollTop = doc.scrollTop || body.scrollTop;
    const scrollHeight = doc.scrollHeight || body.scrollHeight;
    const clientHeight = doc.clientHeight || window.innerHeight;
    const currentDepth = (scrollTop + clientHeight) / scrollHeight;
    return Math.max(0, Math.min(1, currentDepth));
  }

  window.addEventListener('scroll', () => {
    maxScroll = Math.max(maxScroll, getScrollDepth());
  }, { passive: true });

  // Capture clicks relative to the main content canvas
  const canvas = document.getElementById('track-canvas');
  if (canvas) {
    canvas.addEventListener('click', function(ev) {
      const rect = canvas.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;

      fetch('/log_click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          x: x, y: y,
          page_id: canvas.dataset.pageId || 'pageA',
        })
      }).catch(() => {});
    });
  }

  // On unload, send session metrics: time and scroll depth
  function sendMetrics() {
    const duration_sec = (Date.now() - sessionStart) / 1000.0;
    fetch('/log_event', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      keepalive: true,
      body: JSON.stringify({
        type: 'page_end',
        duration_sec: duration_sec,
        max_scroll_depth: maxScroll
      })
    }).catch(() => {});
  }

  window.addEventListener('beforeunload', sendMetrics);
})();
