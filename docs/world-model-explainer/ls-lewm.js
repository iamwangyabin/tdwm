(() => {
  const links = [...document.querySelectorAll('.top-nav a')];
  const sections = [...document.querySelectorAll('main section[id]')];

  const setActive = (id) => {
    links.forEach((link) => {
      link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
    });
  };

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActive(visible.target.id);
    }, { rootMargin: '-25% 0px -60% 0px', threshold: [0, .25, .6, 1] });
    sections.forEach((section) => observer.observe(section));
  }

  const mathScript = document.querySelector('script[src*="mathjax"]');
  const markMathReady = () => document.documentElement.classList.add('math-ready');
  if (window.MathJax?.startup?.promise) {
    window.MathJax.startup.promise.then(markMathReady);
  } else if (mathScript) {
    mathScript.addEventListener('load', () => {
      window.MathJax?.startup?.promise?.then(markMathReady);
    });
  }
})();
