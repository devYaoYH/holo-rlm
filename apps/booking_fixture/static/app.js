(async () => {
  const config = await fetch('/api/config', {cache: 'no-store'}).then(r => r.json());
  document.getElementById('seed').textContent = config.seed;
  const results = document.getElementById('results');
  for (const hotel of config.hotels) {
    const card = document.createElement('article');
    card.className = 'hotel-card';
    card.id = `hotel-${hotel.id}`;
    card.dataset.test = `hotel-${hotel.id}`;
    card.innerHTML = `
      <div class="hotel-image" style="--c1:${hotel.colors[0]};--c2:${hotel.colors[1]}">${hotel.initials}</div>
      <div class="hotel-copy"><h2>${hotel.name}</h2><p>${hotel.description}</p><div class="chips"><span class="chip">Free cancellation</span><span class="chip">Local fixture</span></div></div>
      <div class="hotel-action"><div class="rating">${hotel.rating} / 10</div><div class="price">$${hotel.price}<small>3 nights, fixture price</small></div><a class="button" data-test="details-link-${hotel.id}" href="/details/${hotel.id}">View details</a></div>`;
    results.appendChild(card);
  }
  const update = () => {
    document.querySelector('[data-test=fixture-seed]').textContent = `seed=${config.seed}`;
    document.querySelector('[data-test=scroll-position]').textContent = `scrollY=${Math.round(window.scrollY)}`;
    document.querySelector('[data-test=details-open]').textContent = 'details=false';
  };
  let timer;
  addEventListener('scroll', () => {
    update(); clearTimeout(timer);
    timer = setTimeout(() => fetch('/api/scroll', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({scroll_y:Math.round(window.scrollY)})}), 40);
  }, {passive:true});
  update();
})();
