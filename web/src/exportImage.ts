import { toBlob } from 'html-to-image';

// The DOM classes that are part of the live UI but not the board picture: the
// action buttons, the status line, and the keyboard cursor hint. Filtered out
// of any exported image so a capture shows only the position, not the controls.
const EXCLUDED_CLASSES = ['action-bar', 'manual-status', 'cursor-hint'];

// Render a layout node (board + racks + whatever it wraps) to a PNG blob that
// matches the on-screen look: retina-scaled, on the page's own background, with
// the control chrome filtered out. Shared by the manual UI's "Export PNG"
// button and the offline render harness so both produce identical images.
export function captureLayoutBlob(node: HTMLElement): Promise<Blob | null> {
  return toBlob(node, {
    pixelRatio: 2,
    backgroundColor: getComputedStyle(document.body).backgroundColor || '#1a2632',
    filter: (el: HTMLElement) =>
      !EXCLUDED_CLASSES.some((c) => el.classList?.contains(c)),
  });
}

// Same capture as a base64 data URL, for callers (e.g. a headless render driver)
// that need the bytes as a string rather than a Blob.
export async function captureLayoutDataUrl(node: HTMLElement): Promise<string> {
  const blob = await captureLayoutBlob(node);
  if (!blob) throw new Error('render failed');
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('failed to read blob'));
    reader.readAsDataURL(blob);
  });
}
