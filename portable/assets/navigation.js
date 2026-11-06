/* Reusable view shell: routes come from data-route / data-page, no backend coupling. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const links = [...document.querySelectorAll('#main-nav [data-route]')];
  const panels = [...document.querySelectorAll('[data-page]')];
  const routes = new Map(links.map(link => [link.dataset.route, link.textContent.trim()]));
  const fallback = links[0]?.dataset.route;
  if (!fallback) return;
  let active = null;
  function drawer(open) {
    document.body.classList.toggle('nav-open', open);
    $('mobile-nav').setAttribute('aria-expanded', String(open));
    $('mobile-nav').setAttribute('aria-label', open ? '收起导航' : '展开导航');
    $('nav-backdrop').hidden = !open;
    const narrow = matchMedia('(max-width: 899px)').matches;
    $('sidebar').inert = narrow && !open;
  }
  function paint(route, focus = true) {
    active = route;
    for (const panel of panels) panel.hidden = panel.dataset.page !== route;
    for (const link of links) {
      const selected = link.dataset.route === route;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    $('page-title').textContent = routes.get(route);
    document.title = routes.get(route) + ' · AstroGraph';
    drawer(false);
    if (focus) {
      const heading = document.querySelector(`[data-page="${route}"] h1`);
      heading?.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: 'auto' });
    }
    document.dispatchEvent(new CustomEvent('astrograph:routechange', {detail: {route}}));
  }
  function navigate(route, options = {}) {
    route = routes.has(route) ? route : fallback;
    if (location.hash !== '#' + route) {
      if (options.replace) history.replaceState(null, '', '#' + route);
      else location.hash = route;
    }
    if (active !== route) paint(route, options.focus !== false);
    else drawer(false);
  }
  for (const link of links) link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); navigate(link.dataset.route);
  });
  document.addEventListener('click', event => {
    const jump = event.target.closest('[data-view-jump]');
    if (jump) { event.preventDefault(); navigate(jump.dataset.viewJump); }
  });
  $('mobile-nav').addEventListener('click', () => drawer(!document.body.classList.contains('nav-open')));
  $('nav-backdrop').addEventListener('click', () => { drawer(false); $('mobile-nav').focus(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('nav-open')) {
      drawer(false); $('mobile-nav').focus();
    }
  });
  window.addEventListener('hashchange', () => {
    // The skip link must remain an in-page accessibility action, not an unknown route.
    if (location.hash === '#content') { $('content').focus(); return; }
    navigate(location.hash.slice(1), {replace: true});
  });
  matchMedia('(max-width: 899px)').addEventListener('change', () => drawer(false));
  window.AstroGraphShell = Object.freeze({navigate, get current() { return active; }});
  navigate(location.hash.slice(1), {replace: true, focus: false});
})();
