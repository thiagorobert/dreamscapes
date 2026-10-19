import './styles.css';
import type { Manifest } from './types';

// Scaffold entry point: proves the site can load the built manifest. The real experience
// (stage, hotspots, transitions, hints) replaces this.

async function loadManifest(): Promise<Manifest> {
  const response = await fetch(`${import.meta.env.BASE_URL}scenes.json`);
  if (!response.ok) {
    throw new Error(`Could not load scenes.json (HTTP ${String(response.status)})`);
  }
  return (await response.json()) as Manifest;
}

async function render(root: HTMLElement): Promise<void> {
  const heading = document.createElement('h1');
  heading.textContent = 'Dreamscapes';
  const subtitle = document.createElement('p');
  root.replaceChildren(heading, subtitle);

  const manifest = await loadManifest();
  const start = manifest.scenes.find((scene) => scene.id === manifest.start);
  subtitle.textContent = start?.title ?? '';
}

const root = document.querySelector<HTMLElement>('#app');
if (root) {
  render(root).catch((error: unknown) => {
    console.error(error);
  });
}
