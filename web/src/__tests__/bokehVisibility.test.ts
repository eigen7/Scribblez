import { describe, expect, it } from 'vitest';
import { applyVisibility, NamedModel } from '../lib/bokehVisibility';

// A model that counts writes to `visible`, so a no-op application is observable.
class StubModel implements NamedModel {
  writes = 0;
  private shown = true;
  get visible() { return this.shown; }
  set visible(v: boolean) { this.shown = v; this.writes += 1; }
}

function stubDoc(names: string[]) {
  const models: Record<string, StubModel> = {};
  for (const n of names) models[n] = new StubModel();
  return { models, get_model_by_name: (n: string) => models[n] ?? null };
}

describe('applyVisibility', () => {
  it('sets each named model to the requested visibility', () => {
    const doc = stubDoc(['x_linear', 'x_log']);
    applyVisibility(doc, { x_linear: false, x_log: true });
    expect(doc.models.x_linear.visible).toBe(false);
    expect(doc.models.x_log.visible).toBe(true);
  });

  it('does not write a model whose visibility already matches, and skips unknown names', () => {
    const doc = stubDoc(['x_linear']);
    applyVisibility(doc, { x_linear: true, missing: false });
    expect(doc.models.x_linear.writes).toBe(0);
  });

  it('tolerates a missing document', () => {
    expect(() => applyVisibility(null, { x_log: true })).not.toThrow();
  });
});
