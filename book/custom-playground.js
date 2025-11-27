(function () {
    // Override the playground URL used by mdbook
    // mdbook's default book.js often uses a variable or hardcoded string.
    // However, many themes expose a configuration object.

    // Attempt to intercept fetch requests to play.rust-lang.org
    const originalFetch = window.fetch;
    window.fetch = function (url, options) {
        if (typeof url === 'string' && url.includes('play.rust-lang.org')) {
            console.log("Intercepting playground request to local server");
            // Redirect to local server
            // The local server expects /evaluate.json
            return originalFetch("http://localhost:3001/evaluate.json", options);
        }
        return originalFetch(url, options);
    };

    // Intercept playground requests
    window.playground_url = 'http://localhost:3001/evaluate.json';

    // Forcefully unhide play buttons using MutationObserver
    // This handles cases where buttons are added dynamically or hidden by other scripts
    var observer = new MutationObserver(function (mutations) {
        document.querySelectorAll('.play-button.hidden').forEach(function (button) {
            button.classList.remove('hidden');
            button.removeAttribute('hidden');
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'hidden']
    });

    // Also run once immediately
    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('.play-button.hidden').forEach(function (button) {
            button.classList.remove('hidden');
            button.removeAttribute('hidden');
        });
    });

    console.log("Local playground interceptor loaded.");
})();
