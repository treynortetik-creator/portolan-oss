// Shared HTML-escaping helper for Portolan's JS islands.
//
// Review finding 22 (grouped nits): chat.html and graph.html each
// defined their own escapeHtml -- one DOM-based (a throwaway <div>'s
// textContent/innerHTML round-trip), one a plain regex -- doing the same
// job slightly differently, now that there are two islands sharing one
// file is simpler than two divergent inline copies. Every call site in
// both templates only ever inserts the result into an element's content
// (never into an attribute value), so escaping the five characters below
// is sufficient in both places; the regex form was picked as the shared
// implementation since it needs no DOM node and already covered quotes,
// which the content-position call sites never needed but which are
// harmless to escape anyway.
//
// Not a module (no bundler in this project, plain <script> tags only) --
// a single global, namespaced so it can't collide with anything a future
// third-party script defines.
(function (global) {
  'use strict';
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  global.PortolanEscapeHtml = escapeHtml;
})(window);
