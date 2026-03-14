/* ele — result page interactions */
/* Before/After comparison slider using Pointer Events API */

(function () {
  'use strict';

  function initSlider(slider) {
    var proc   = slider.querySelector('.ba-img-proc');
    var handle = slider.querySelector('.ba-handle');
    if (!proc || !handle) return;

    var dragging = false;

    function setPosition(clientX) {
      var rect = slider.getBoundingClientRect();
      var pct  = (clientX - rect.left) / rect.width * 100;
      pct = Math.max(2, Math.min(98, pct));
      proc.style.clipPath = 'inset(0 0 0 ' + pct.toFixed(1) + '%)';
      handle.style.left   = pct.toFixed(1) + '%';
    }

    slider.addEventListener('pointerdown', function (e) {
      dragging = true;
      slider.setPointerCapture(e.pointerId);
      setPosition(e.clientX);
      e.preventDefault();
    });

    slider.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      setPosition(e.clientX);
    });

    slider.addEventListener('pointerup',     function () { dragging = false; });
    slider.addEventListener('pointercancel', function () { dragging = false; });

    /* Keyboard accessibility: left/right arrows nudge the slider */
    slider.setAttribute('tabindex', '0');
    slider.addEventListener('keydown', function (e) {
      var rect  = slider.getBoundingClientRect();
      var cur   = parseFloat(handle.style.left) || 50;
      var delta = (e.shiftKey ? 10 : 2);
      if (e.key === 'ArrowLeft')  { setPosition(rect.left + (cur - delta) / 100 * rect.width); e.preventDefault(); }
      if (e.key === 'ArrowRight') { setPosition(rect.left + (cur + delta) / 100 * rect.width); e.preventDefault(); }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.ba-slider').forEach(initSlider);
  });

}());
