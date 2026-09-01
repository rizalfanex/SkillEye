/**
 * Shared site navigation. Every page drops in:
 *
 *   <div id="siteNav"></div>
 *   <script src="js/skilleye-nav.js"></script>
 *
 * right where the <nav> used to be. This file is the single source of truth
 * for the nav bar, so a nav change only touches this one file.
 *
 * The active link is highlighted from the current URL, so pages do not need
 * any per-page markup. Each page keeps its own language-toggle and mobile-menu
 * wiring -- those elements exist as soon as this script runs during parsing.
 */
(function () {
    var NAV = ''
        + '<nav>'
        + '    <div class="container">'
        + '        <a href="index.html" class="nav-logo">🎾 AI Tennis Coach</a>'
        + '        <ul class="nav-links" id="navLinks">'
        + '            <li><a href="index.html" class="lang-zh">首頁</a><a href="index.html" class="lang-en lang-hidden">Home</a></li>'
        + '            <li><a href="about.html" class="lang-zh">關於</a><a href="about.html" class="lang-en lang-hidden">About</a></li>'
        + '            <li><a href="training.html" class="lang-zh">訓練</a><a href="training.html" class="lang-en lang-hidden">Training</a></li>'
        + '            <li><a href="gear.html" class="lang-zh">裝備</a><a href="gear.html" class="lang-en lang-hidden">Gear</a></li>'
        + '            <li><a href="services.html" class="lang-zh">服務</a><a href="services.html" class="lang-en lang-hidden">AI</a></li>'
        + '            <li><a href="contact.html" class="lang-zh">聯絡</a><a href="contact.html" class="lang-en lang-hidden">Contact</a></li>'
        + '            <li><button class="lang-toggle" id="langToggle">EN</button></li>'
        + '        </ul>'
        + '        <button class="mobile-menu-btn" id="mobileMenuBtn">☰</button>'
        + '    </div>'
        + '</nav>';

    function currentPage() {
        var p = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
        return p;
    }

    function mount() {
        var host = document.getElementById('siteNav');
        if (!host) return;
        host.outerHTML = NAV;
        var key = currentPage();
        var links = document.querySelectorAll('#navLinks a');
        for (var i = 0; i < links.length; i++) {
            var href = (links[i].getAttribute('href') || '').split('/').pop().toLowerCase();
            if (href && href === key) links[i].classList.add('active');
        }
    }

    mount();
    window.SkillEyeNav = { mount: mount };
})();
