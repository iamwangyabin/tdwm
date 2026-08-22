const horizonInput = document.querySelector('#horizon');
const horizonOutput = document.querySelector('#horizon-value');
const successorEquation = document.querySelector('#successor-equation');
const costEquation = document.querySelector('#cost-equation');
const latentSteps = [...document.querySelectorAll('.latent-step')];
const timelineLinks = [...document.querySelectorAll('.timeline-link')];

const powerLabel = (power) => {
  if (power === 0) return '';
  if (power === 1) return 'γ';
  return `γ<sup>${power}</sup>`;
};

const successorTerm = (step) => {
  const weight = powerLabel(step - 1);
  return `${weight}ψ̂<sub>t+${step}</sub>`;
};

const renderHorizon = () => {
  const horizon = Number(horizonInput.value);
  horizonOutput.value = `h = ${horizon}`;
  horizonOutput.textContent = `h = ${horizon}`;

  latentSteps.forEach((step) => {
    step.classList.toggle('active', Number(step.dataset.step) <= horizon);
  });

  timelineLinks.forEach((link) => {
    link.classList.toggle('active', Number(link.dataset.link) < horizon);
  });

  const terms = Array.from({ length: horizon }, (_, index) => successorTerm(index + 1));
  successorEquation.innerHTML = `Ŝ<sub>${horizon}</sub> = ${terms.join(' + ')}`;
  costEquation.innerHTML = `Ĉ<sub>${horizon}</sub>(A,g) = 1 - Ŝ<sub>${horizon}</sub><sup>T</sup>ψ(g) / Z<sub>${horizon}</sub>`;
};

horizonInput.addEventListener('input', renderHorizon);
renderHorizon();
