const container = document.querySelector('.container');
const divider = document.querySelector('.divider');
const leftGif = document.querySelector('.gif:first-child');
const rightGif = document.querySelector('.gif:last-child');
let isDragging = false;

divider.addEventListener('mousedown', function(e) {
  isDragging = true;
});

container.addEventListener('mousemove', function(e) {
  if (!isDragging) return;
  const containerRect = container.getBoundingClientRect();
  const mousePosition = e.clientX - containerRect.left;
  const containerWidth = containerRect.width;
  const dividerWidth = divider.offsetWidth;
  const minLeft = 0;
  const maxLeft = containerWidth - dividerWidth;
  const newLeft = Math.max(minLeft, Math.min(maxLeft, mousePosition));
  const leftWidth = newLeft / containerWidth * 100;
  const rightWidth = 100 - leftWidth;

  divider.style.left = leftWidth + '%';
  leftGif.style.width = leftWidth + '%';
  rightGif.style.left = leftWidth + '%';
  rightGif.style.width = rightWidth + '%';
});

document.addEventListener('mouseup', function(e) {
  isDragging = false;
});
